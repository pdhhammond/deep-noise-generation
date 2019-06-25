"""
From https://github.com/fangchangma/sparse-to-dense.pytorch
"""

import os
import os.path
import numpy as np
import torch.utils.data as data
import h5py
import dataloaders.transforms as transforms

import time
import cv2

import random

IMG_EXTENSIONS = ['.h5',]

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
	classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
	classes.sort()
	class_to_idx = {classes[i]: i for i in range(len(classes))}
	return classes, class_to_idx

def make_dataset(dir, class_to_idx, limit=-1):
	images = []
	dir = os.path.expanduser(dir)

	for i, target in enumerate(sorted(os.listdir(dir))):
		if limit > -1 and i >= limit:
			break

		d = os.path.join(dir, target)
		if not os.path.isdir(d):
			continue
		for root, _, fnames in sorted(os.walk(d)):
			for fname in sorted(fnames):
				if is_image_file(fname):
					path = os.path.join(root, fname)
					item = (path, class_to_idx[target])
					images.append(item)
	return images

def h5_loader(path):
	h5f = h5py.File(path, "r")
	rgb = np.array(h5f['rgb'])
	rgb = np.transpose(rgb, (1, 2, 0))
	depth = np.array(h5f['depth'])
	raw_depth = None if 'rawDepth' not in h5f else np.array(h5f['rawDepth'])
	return rgb, depth, raw_depth

# I need to load gt depths as the raw depth while training
# so I can apply noise to them without the raw depth noise,
# but the true raw depths for validation
class TempLoader(object):
	def __init__(self, type, loader):
		self.type = type
		self.loader = loader

	def __call__(self, path):
		rgb, depth, raw_depth = self.loader(path)
		if self.type == 'train':
			return rgb, depth, depth
		return rgb, depth, raw_depth

# def rgb2grayscale(rgb):
#     return rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114

to_tensor = transforms.ToTensor()

class RawDataloader(data.Dataset):
	modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'
	color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

	def __init__(self, root, type, sparsifier=None, modality='rgb', loader=h5_loader, aug_limit=-1):
		classes, class_to_idx = find_classes(root)
		# raw_classes, raw_class_to_idx = find_classes(raw_root)

		imgs = make_dataset(root, class_to_idx, aug_limit)
		# raw_imgs = make_dataset(raw_root, class_to_idx)

		assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
		print("Found {} images in {} folder.".format(len(imgs), type))
		# assert len(raw_imgs)>0, "Found 0 images in subfolders of: " + raw_root + "\n"
		# print("Found {} images in {} folder.".format(len(raw_imgs), type))

		self.root = root
		self.imgs = imgs
		self.classes = classes
		self.class_to_idx = class_to_idx

		# self.raw_root = raw_root
		# self.raw_imgs = raw_imgs
		# self.raw_classes = raw_classes
		# self.raw_class_to_idx = raw_class_to_idx
		if type == 'train':
			self.transform = self.train_transform
		elif type == 'val':
			self.transform = self.val_transform
		else:
			raise (RuntimeError("Invalid dataset type: " + type + "\n"
								"Supported dataset types are: train, val"))
		self.loader = loader
		# self.loader = TempLoader(type, loader)
		self.sparsifier = sparsifier

		assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
								"Supported dataset types are: " + ''.join(self.modality_names)
		self.modality = modality

	def train_transform(self, rgb, depth):
		raise (RuntimeError("train_transform() is not implemented. "))

	def val_transform(rgb, depth):
		raise (RuntimeError("val_transform() is not implemented."))

	def create_sparse_depth(self, rgb, depth):
		if self.sparsifier is None:
			return depth
		else:
			mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
			sparse_depth = np.zeros(depth.shape)

			# if a residual map has been passed
			if isinstance(mask_keep, tuple):
				mask_keep, noise_offset = mask_keep
				noise_offset *= 10.
				depth = np.clip(depth + noise_offset, 0., 10.)

			sparse_depth[mask_keep] = depth[mask_keep]
			return sparse_depth

	def create_rgbd(self, rgb, depth):
		sparse_depth = self.create_sparse_depth(rgb, depth)
		rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
		return rgbd

	def __getraw__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (rgb, depth) the raw data.
		"""
		path, target = self.imgs[index]
		rgb, depth, raw_depth = self.loader(path)

		# raw_path, raw_target = self.raw_imgs[index]
		# _, raw_depth, _ = self.loader(raw_path)

		return rgb, depth, raw_depth

	def __getitem__(self, index):
		rgb, depth, raw_depth = self.__getraw__(index)
		# pick_real = random.randint(0, 10) == 0
		# sparse_depth = self.create_sparse_depth(rgb, depth) if not pick_real else raw_depth
		sparse_depth = raw_depth if self.sparsifier is None else self.create_sparse_depth(rgb, depth)

		if self.transform is not None:
			seed = int(time.time())
			rgb_np, depth_np = self.transform(rgb, depth, seed)
			_, sparse_depth_np = self.transform(rgb, sparse_depth, seed)
		else:
			raise(RuntimeError("transform not defined"))

		if self.modality == 'rgb':
			input_np = rgb_np
		elif self.modality == 'rgbd':
			# input_np = self.create_rgbd(rgb_np, raw_depth_np)
			input_np = np.append(rgb_np, np.expand_dims(sparse_depth_np, axis=2), axis=2)
		elif self.modality == 'd':
			# input_np = self.create_sparse_depth(rgb_np, raw_depth_np)
			input_np = sparse_depth_np

		input_tensor = to_tensor(input_np)
		while input_tensor.dim() < 3:
			input_tensor = input_tensor.unsqueeze(0)
		depth_tensor = to_tensor(depth_np)
		depth_tensor = depth_tensor.unsqueeze(0)

		return input_tensor, depth_tensor

	def __len__(self):
		return len(self.imgs)

class MyDataloader(data.Dataset):
	modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'
	color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

	def __init__(self, root, type, sparsifier=None, modality='rgb', loader=h5_loader):
		classes, class_to_idx = find_classes(root)
		imgs = make_dataset(root, class_to_idx)
		assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
		print("Found {} images in {} folder.".format(len(imgs), type))
		self.root = root
		self.imgs = imgs
		self.classes = classes
		self.class_to_idx = class_to_idx
		if type == 'train':
			self.transform = self.train_transform
		elif type == 'val':
			self.transform = self.val_transform
		else:
			raise (RuntimeError("Invalid dataset type: " + type + "\n"
								"Supported dataset types are: train, val"))
		self.loader = loader
		self.sparsifier = sparsifier

		assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
								"Supported dataset types are: " + ''.join(self.modality_names)
		self.modality = modality

	def train_transform(self, rgb, depth):
		raise (RuntimeError("train_transform() is not implemented. "))

	def val_transform(rgb, depth):
		raise (RuntimeError("val_transform() is not implemented."))

	def create_sparse_depth(self, rgb, depth):
		if self.sparsifier is None:
			return depth
		else:
			mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
			sparse_depth = np.zeros(depth.shape)

			# if a residual map has been passed
			if isinstance(mask_keep, tuple):
				mask_keep, noise_offset = mask_keep

				depth = np.clip((depth / 10.) + noise_offset, 0., 10.)
				depth = depth * 10

			sparse_depth[mask_keep] = depth[mask_keep]
			return sparse_depth

	def create_rgbd(self, rgb, depth):
		sparse_depth = self.create_sparse_depth(rgb, depth)
		rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
		return rgbd

	def __getraw__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (rgb, depth) the raw data.
		"""
		path, target = self.imgs[index]
		rgb, depth, raw_depth = self.loader(path)
		return rgb, depth

	def __getitem__(self, index):
		rgb, depth = self.__getraw__(index)
		if self.transform is not None:
			rgb_np, depth_np = self.transform(rgb, depth)
		else:
			raise(RuntimeError("transform not defined"))

		# color normalization
		# rgb_tensor = normalize_rgb(rgb_tensor)
		# rgb_np = normalize_np(rgb_np)

		if self.modality == 'rgb':
			input_np = rgb_np
		elif self.modality == 'rgbd':
			input_np = self.create_rgbd(rgb_np, depth_np)
		elif self.modality == 'd':
			input_np = self.create_sparse_depth(rgb_np, depth_np)

		input_tensor = to_tensor(input_np)
		while input_tensor.dim() < 3:
			input_tensor = input_tensor.unsqueeze(0)
		depth_tensor = to_tensor(depth_np)
		depth_tensor = depth_tensor.unsqueeze(0)

		return input_tensor, depth_tensor

	def __len__(self):
		return len(self.imgs)

	# def __get_all_item__(self, index):
	#     """
	#     Args:
	#         index (int): Index

	#     Returns:
	#         tuple: (input_tensor, depth_tensor, input_np, depth_np)
	#     """
	#     rgb, depth = self.__getraw__(index)
	#     if self.transform is not None:
	#         rgb_np, depth_np = self.transform(rgb, depth)
	#     else:
	#         raise(RuntimeError("transform not defined"))

	#     # color normalization
	#     # rgb_tensor = normalize_rgb(rgb_tensor)
	#     # rgb_np = normalize_np(rgb_np)

	#     if self.modality == 'rgb':
	#         input_np = rgb_np
	#     elif self.modality == 'rgbd':
	#         input_np = self.create_rgbd(rgb_np, depth_np)
	#     elif self.modality == 'd':
	#         input_np = self.create_sparse_depth(rgb_np, depth_np)

	#     input_tensor = to_tensor(input_np)
	#     while input_tensor.dim() < 3:
	#         input_tensor = input_tensor.unsqueeze(0)
	#     depth_tensor = to_tensor(depth_np)
	#     depth_tensor = depth_tensor.unsqueeze(0)

	#     return input_tensor, depth_tensor, input_np, depth_np
