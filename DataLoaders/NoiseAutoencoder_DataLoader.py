"""
Data loader for depth noise dataset to train noise autoencoder.
"""

import os
import cv2
import random

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


folder_seed = 1000

def partition_dataset(root_path, train_cap, val_cap, t_sample_cap,
						v_sample_cap):
	folders = next(os.walk(root_path))[1]
	folders.sort()
	t_descriptor, v_descriptor = [], []
	scene_cap = train_cap + val_cap

	# shuffle and sample the folders as indicated
	random.seed(folder_seed)
	random.shuffle(folders)
	scene_cap = len(folders) if scene_cap < 0 else scene_cap
	folders = folders[: min(scene_cap, len(folders))]

	for i, folder in enumerate(folders):
		sample_cap = t_sample_cap if i < train_cap else v_sample_cap
		desc_list = t_descriptor if i < train_cap else v_descriptor

		noise_path = os.path.join(root_path, folder, "noise")
		samples = len(next(os.walk(noise_path))[2])
		num_samples = min(samples, samples if sample_cap < 0 else sample_cap)

		scene_descriptor = [(folder, j) for j in range(0, num_samples)]
		desc_list += scene_descriptor

	return t_descriptor, v_descriptor

def build_descriptor(root_path):
	folders = next(os.walk(root_path))[1]
	descriptor = []

	for folder in folders:
		noise_path = os.path.join(root_path, folder, "noise")
		num_samples = len(next(os.walk(noise_path))[2])

		scene_descriptor = [(folder, j) for j in range(0, num_samples)]
		descriptor += [scene_descriptor]
	return descriptor

class DepthNoiseDataset(Dataset):
	"""Depth Noise dataset."""

	def __init__(self, root_dir, descriptor, sample_cap, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.sample_cap = sample_cap

		# if descriptor is not None else build_descriptor(root_dir)
		self.descriptor = descriptor

	def __len__(self):
		return len(self.descriptor)

	def get(self, indices):
		samples = [self[idx] for idx in indices]

		image = torch.cat([samples[i]["image"] for i in range(len(samples))], 0)
		depth = torch.cat([samples[i]["depth"] for i in range(len(samples))], 0)
		dropout = torch.cat([samples[i]["dropout"] for i in range(len(samples))], 0)
		noise = torch.cat([samples[i]["noise"] for i in range(len(samples))], 0)

		return {"image":image, "depth": depth, "dropout":dropout, "noise":noise}

	def __getitem__(self, idx):
		# restrict the amount of possible input
		folder, sample_idx = self.descriptor[idx]

		# get scene directory name
		scene_dir = os.path.join(self.root_dir, folder)

		rgb_name = os.path.join(scene_dir, "med_image_%03d.png")
		rgb_name = rgb_name % self.sample_cap
		d_name = os.path.join(scene_dir, "med_depth_%03d.png")
		rgb_name = d_name % self.sample_cap
		raw_name = os.path.join(scene_dir, "depths", "depth_%04d.npy")
		raw_name = raw_name % sample_idx

		raw_depth = np.clip(np.load(raw_name) / 10000., 0., 1.).astype(np.float64)
		med_depth = cv2.imread(d_name, 0).astype(np.float64) / 255.

		dropout = np.ones_like(raw_depth)
		dropout[raw_depth == 0] = 0
		dropout = (dropout * 255.).astype(np.uint8)

		noise = raw_depth - med_depth
		noise[dropout == 0] = 0
		noise = ((noise + 1.) / 2. * 255.).astype(np.uint8)

		# LOADING IMAGES IN BLACK AND WHITE
		sample = {"image": cv2.imread(rgb_name, 0),
					"depth": cv2.imread(d_name, 0),
					"dropout": dropout, "noise": noise}

		if self.transform:
			sample = self.transform(sample)
		return sample


class Rescale(object):
	"""Rescale the image in a sample to a given size.

	Args:
		output_size (tuple or int): Desired output size. If tuple, output is
			matched to output_size. If int, smaller of image edges is matched
			to output_size keeping aspect ratio the same.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, depth, dropout, noise = sample['image'], sample['depth'], sample['dropout'], sample['noise']

		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		image = cv2.resize(image.astype(np.float64) / 255., (new_h, new_w))
		depth = cv2.resize(depth.astype(np.float64) / 255, (new_h, new_w))
		dropout = cv2.resize(dropout.astype(np.float64) / 255., (new_h, new_w), interpolation=cv2.INTER_NEAREST)
		noise = cv2.resize(noise.astype(np.float64) / 255, (new_h, new_w))

		return {'image': image, 'depth': depth, 'dropout': dropout, 'noise': noise}


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, depth, dropout, noise = sample['image'], sample['depth'], sample['dropout'], sample['noise']

		# image = torch.from_numpy(image.transpose((2, 0, 1)))
		image = torch.from_numpy(np.expand_dims(image, axis=0))
		depth = torch.from_numpy(np.expand_dims(depth, axis=0))
		dropout = torch.from_numpy(np.expand_dims(dropout, axis=0))
		noise = torch.from_numpy(np.expand_dims(noise, axis=0))
		return {'image': image, 'depth': depth, 'dropout': dropout, 'noise': noise}

class Normalize(object):
	def __call__(self, sample):
		image, depth, dropout, noise = sample['image'], sample['depth'], sample['dropout'], sample['noise']
		norm_3c = transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5])
		norm_1c = transforms.Normalize(mean=[.5], std=[.5])

		image = norm_1c(image.float())
		depth = norm_1c(depth.float())
		dropout = norm_1c(dropout.float())
		noise = norm_1c(noise.float())
		return {'image': image, 'depth': depth, 'dropout': dropout, 'noise': noise}

class SoftLabels(object):
	def __call__(self, sample):
		dropout = sample['dropout']

		noise_image = torch.zeros_like(dropout).uniform_(0.02, .1)
		# noise_image = torch.FloatTensor(N, 1, sample.size(2), sample.size(3)).uniform_(0.02, .1)
		dropout[dropout <= 0] += noise_image[dropout <= 0]
		dropout[dropout > 0] -= noise_image[dropout > 0]

		sample['dropout'] = dropout
		return sample

class Crop(object):
	"""Crop numpy images to size."""

	def __init__(self, top, bottom, left, right):
		self.top, self.bottom, self.left, self.right = top, bottom, left, right

	def __call__(self, sample):
		image, depth, dropout, noise = sample['image'], sample['depth'], sample['dropout'], sample['noise']
		image = image[self.top:self.bottom, self.left:self.right]
		depth = depth[self.top:self.bottom, self.left:self.right]
		dropout = dropout[self.top:self.bottom, self.left:self.right]
		noise = noise[self.top:self.bottom, self.left:self.right]

		return {'image': image, 'depth': depth, 'dropout': dropout, 'noise': noise}

def depth_noise_data(size):
	compose = transforms.Compose([
		Rescale(size),
		ToTensor(),
		Normalize()
		])
	out_dir = './dataset'
	return DepthNoiseDataset("small_train.json", "./DepthVids", transform=compose)