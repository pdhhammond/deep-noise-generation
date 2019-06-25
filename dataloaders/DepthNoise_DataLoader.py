"""
Data loader for depth noise dataset.
"""

import os
import cv2
import random

from time import time

import numpy as np
import dataloaders.transforms as transforms

import torch
from torch.utils.data import Dataset, DataLoader


iheight, iwidth = 480, 640 # raw image size
to_tensor = transforms.ToTensor()
folder_seed = 1000

def build_descriptor(root_path, sample_cap, scene_cap, num_augmented):
	'''
	Helper method for DepthNoiseDataset. Builds a descriptor for the dataset
	found at root_path.

	Parameters
	----------
	root_path : str
		Path to the root data directory.
	sample_cap : int
		The number of frames to include from each scene.
	scene_cap : int
		The total number of scenes to include in the dataset.
	num_augmented : int
		The number of augmented frames to produce.

	Returns
	-------
	folders : [str, ...]
		List of folder names for each scene included in the dataset.
	descriptor : [(folder_idx, sample_idx, aug_seed), ...]
		Descriptor for the dataset. A sample_idx of -1 indicates a virtual
		(augmented) frame.
	'''
	counter = 0
	folders = next(os.walk(root_path))[1]
	folders.sort()
	descriptor = []

	# shuffle and sample the folders if indicated
	random.seed(folder_seed)
	random.shuffle(folders)
	scene_cap = len(folders) if scene_cap < 0 else scene_cap
	folders = folders[: min(scene_cap, len(folders))]

	for i, folder in enumerate(folders):
		noise_path = os.path.join(root_path, folder, "noise")
		samples = len(next(os.walk(noise_path))[2])
		num_samples = min(samples, samples if sample_cap < 0 else sample_cap)
		aug_samples = num_augmented

		scene_descriptor = [(i, j, j) for j in range(0, num_samples)]
		augment_descriptor = [(i, -1, counter+j) for j in range(0, aug_samples)]
		scene_descriptor += augment_descriptor

		descriptor += scene_descriptor
		counter += aug_samples

	return folders, descriptor

# gt_depth in range [0, 10]
# noise and dropout in range [0, 1]
# returns raw_depth in range [0, 10]
def make_raw_depth(gt_depth, noise, dropout):
	raw_depth = (gt_depth / 10.) + ((noise * 2.) - 1.)
	raw_depth = np.clip(raw_depth, 0., 1.)
	raw_depth[dropout < 1.] = 0
	return raw_depth * 10.

class DepthNoiseDataset(Dataset):
	color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

	def __init__(self, root_dir, type, sparsifier=None, num_augmented=0,
		sample_cap=-1, scene_cap=-1, sim_offline=False):

		self.root_dir = root_dir
		self.sim_offline = sim_offline
		self.sparsifier = sparsifier

		self.transform = self.val_transform
		if type == "train":
			self.train_transform

		self.folders, self.descriptor = build_descriptor(root_dir, sample_cap,
			scene_cap, num_augmented)
		self.output_size = (228, 304)

		self.type = type
		self.sample_cap = sample_cap

	def __len__(self):
		return len(self.descriptor)

	def create_sparse_depth(self, rgb, depth, seed=None):
		if self.sparsifier is None:
			return depth
		else:
			mask_keep = self.sparsifier.dense_to_sparse(rgb, depth, seed)
			sparse_depth = np.zeros(depth.shape)

			# if a residual map has been passed
			if isinstance(mask_keep, tuple):
				mask_keep, noise_offset = mask_keep
				noise_offset *= 10.
				depth = np.clip(depth + noise_offset, 0., 10.)

			sparse_depth[mask_keep] = depth[mask_keep]
			return sparse_depth

	def __getitem__(self, idx):
		# restrict the amount of possible input
		folder_idx, sample_idx, aug_seed = self.descriptor[idx]

		# get scene directory name
		scene_dir = os.path.join(self.root_dir, self.folders[folder_idx])

		if self.type == "train":
			if self.sample_cap > 0:
				rgb_name = os.path.join(scene_dir, "med_image_%03d.png")
				rgb_name = rgb_name % self.sample_cap
				gt_d_name = os.path.join(scene_dir, "med_depth_%03d.png")
				gt_d_name = gt_d_name % self.sample_cap
			else:
				rgb_name = os.path.join(scene_dir, "med_image.png")
				gt_d_name = os.path.join(scene_dir, "med_depth_filled.png")

			# the sparse to dense method labels pure white depths as dropout
			depth = np.clip(cv2.imread(gt_d_name, 0), 0, 254)
			depth = (depth.astype(np.float64) / 255.) * 10.
		else:
			rgb_name = os.path.join(scene_dir, "med_image.png")
			gt_d_name = os.path.join(scene_dir, "med_depth_filled.npy")
			depth = np.load(gt_d_name) * 10.

		# load images
		rgb = cv2.imread(rgb_name, 1)
		sp_d_name = os.path.join(scene_dir, "depths", "depth_%04d.npy")
		sp_d_name = sp_d_name % sample_idx

		# get sparse depth
		if sample_idx < 0:
			seed = aug_seed if self.sim_offline else None
			sparse_depth = self.create_sparse_depth(rgb, depth, seed)
		else:
			sparse_depth = np.clip(np.load(sp_d_name), 0, 9999).astype('float')
			sparse_depth = sparse_depth / 1000.

		# Applying augmentation
		seed = int(time())
		rgb_np, depth_np = self.transform(rgb, depth, seed)
		_, sparse_depth_np = self.transform(rgb, sparse_depth, seed)

		input_np = np.expand_dims(sparse_depth_np, axis=2)
		input_np = np.append(rgb_np, input_np, axis=2)

		# I don't know what this is for, but I'm running with it
		input_tensor = to_tensor(input_np)
		while input_tensor.dim() < 3:
			input_tensor = input_tensor.unsqueeze(0)
		depth_tensor = to_tensor(depth_np)
		depth_tensor = depth_tensor.unsqueeze(0)

		return input_tensor, depth_tensor

	def train_transform(self, rgb, depth, random_seed):
		np.random.seed(random_seed)

		s = np.random.uniform(1.0, 1.5) # random scaling
		depth_np = depth / s
		angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
		do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

		# perform 1st step of data augmentation
		transform = transforms.Compose([
			# this is for computational efficiency, since rotation can be slow
			transforms.Resize(250.0 / iheight),
			transforms.Rotate(angle),
			transforms.Resize(s),
			transforms.CenterCrop(self.output_size),
			transforms.HorizontalFlip(do_flip)
		])
		rgb_np = transform(rgb)
		rgb_np = self.color_jitter(rgb_np) # random color jittering
		rgb_np = np.asfarray(rgb_np, dtype='float') / 255
		depth_np = transform(depth_np)

		return rgb_np, depth_np

	def val_transform(self, rgb, depth, random_seed):
		np.random.seed(random_seed)

		depth_np = depth
		transform = transforms.Compose([
			transforms.Resize(240.0 / iheight),
			transforms.CenterCrop(self.output_size),
		])
		rgb_np = transform(rgb)
		rgb_np = np.asfarray(rgb_np, dtype='float') / 255
		depth_np = transform(depth_np)

		return rgb_np, depth_np