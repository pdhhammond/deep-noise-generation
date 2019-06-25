import os
import cv2
import torch
import numpy as np

from torch.autograd import Variable

def load_by_index(path, indices, color_mode=0):
	if not os.path.isdir(path):
		return None

	files = next(os.walk(path))[2]
	files.sort()

	image_stack = []
	filenames = []

	for name in files:
		id = int(name.split("_")[-1].split(".")[0])
		if id not in indices: continue

		filenames += [path + name]
		image_stack += [load_image(filenames[-1], color_mode)]

	return np.array(image_stack), filenames


def load_image_stack(path, color_mode=0):
	if not os.path.isdir(path):
		return None

	files = next(os.walk(path))[2]
	files.sort()

	image_stack = []
	filenames = []

	for i in range(len(files)):
		filenames += [os.path.join(path, files[i])]
		image_stack += [load_image(filenames[-1], color_mode)]

	return np.array(image_stack), filenames

def load_image(path, color_mode):
	if has_extension(path, "jpg") or has_extension(path, "png"):
		return cv2.imread(path, color_mode)
	if has_extension(path, "npy"):
		return np.load(path)
	return None

def has_extension(path, ext):
	extension = path.split(".")[-1]
	return ext.split(".")[-1] == extension

def normalize_depth(raw, cutoff=10000):
	return np.clip(raw / cutoff, 0., 1.)
