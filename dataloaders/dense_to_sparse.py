"""
From https://github.com/fangchangma/sparse-to-dense.pytorch
Modified to work with our data augmentation method and dense noisy RGB-D images.
"""

import numpy as np
import cv2

import torch
from torch import nn
from torch.autograd.variable import Variable

from models import VariationalDecoderNet
from torchvision import transforms

from dataloaders.noise_routines import get_naive_noise


def rgb2grayscale(rgb):
	return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


class DenseToSparse:
	def __init__(self):
		pass

	def dense_to_sparse(self, rgb, depth):
		pass

	def __repr__(self):
		pass

class AlgorithmicNoise(DenseToSparse):
	name = "algorithmic"

	def __init__(self):
		DenseToSparse.__init__(self)

	def __repr__(self):
		return "AlgorithmicNoise"

	def dense_to_sparse(self, rgb, depth, seed=None):
		h, w = rgb.shape[0], rgb.shape[1]
		rgb = cv2.cvtColor(rgb[..., ::-1], cv2.COLOR_RGB2GRAY)
		rgb, depth = (cv2.pyrDown(rgb), cv2.pyrDown(depth/10))

		if seed is not None:
			np.random.seed(seed)

		dropout, noise = get_naive_noise(rgb, depth)

		dropout = cv2.resize(dropout, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255
		dropout = dropout.astype(np.bool)

		noise = (np.clip((noise + 1.) / 2., 0, 1) * 255).astype(np.uint8)
		noise = (cv2.resize(noise, (w, h)).astype(np.float32) / 255.) * 2 - 1

		return dropout, noise

class SimulatedCameraNoise(DenseToSparse):
	name = "sim_camera"

	def __init__(self, filename, z_dims=100):
		DenseToSparse.__init__(self)
		self.generator = VariationalDecoderNet(z_dims)
		# self.generator.cuda()

		weights = torch.load(filename)
		self.generator.load_state_dict(weights)
		# self.generator.eval()

		self.norm = transforms.Normalize(mean=[.5], std=[.5])

	def __repr__(self):
		return "IAMTHEMONKEYNERD"

	def noise(self, size):
		'''
		Generates a 1-d vector of gaussian-sampled random values
		'''
		n = torch.randn(size, 100, 1, 1)
		n = Variable(nn.functional.normalize(n, p=2, dim=1))
		return n

	def dense_to_sparse(self, rgb, depth, seed=None):
		h, w = rgb.shape[0], rgb.shape[1]

		# process rgb
		rgb = cv2.cvtColor(rgb.astype(np.uint8)[..., ::-1], cv2.COLOR_RGB2GRAY)
		rgb = cv2.resize(rgb.astype(np.float64) / 255., (256, 256)) * 2. - 1.
		rgb = torch.from_numpy(np.expand_dims(rgb, axis=0)).float()
		rgb = torch.unsqueeze(self.norm(rgb).float(), 1)

		# process depth
		depth = depth.astype(np.float64) / 10.
		depth = cv2.resize(depth, (256, 256)) * 2. - 1.
		depth_resized = depth
		depth = torch.from_numpy(np.expand_dims(depth, axis=0)).float()
		depth = torch.unsqueeze(self.norm(depth).float(), 1)

		label = Variable(torch.cat([rgb, depth], 1))

		# generate noise
		if seed is not None:
			torch.manual_seed(seed)

		generated = self.generator.generate(self.noise(1), label)
		generated = generated.data.cpu().numpy()

		# post-process dropout
		gen_dropout = (generated[0, 0] + 1.) / 2.
		gen_dropout = cv2.resize(gen_dropout, (w, h)) * 2. - 1.

		# prepare dropout boolean mask
		gen_dropout[gen_dropout <= 0] = 0
		gen_dropout[gen_dropout > 0] = 1
		gen_dropout = gen_dropout.astype(np.bool)

		# post-process depth-noise
		gen_depth_noise = (generated[0, 1] + 1.) / 2.
		gen_depth_noise = (gen_depth_noise * 255.).astype(np.uint8)
		gen_depth_noise = cv2.medianBlur(gen_depth_noise, 7)
		gen_depth_noise = cv2.resize(gen_depth_noise, (w, h)).astype(np.float32) / 255.
		gen_depth_noise = gen_depth_noise * 2 - 1

		return gen_dropout, gen_depth_noise


class UniformSampling(DenseToSparse):
	name = "uar"
	def __init__(self, num_samples, max_depth=np.inf):
		DenseToSparse.__init__(self)
		self.num_samples = num_samples
		self.max_depth = max_depth

	def __repr__(self):
		return "%s{ns=%d,md=%f}" % (self.name, self.num_samples, self.max_depth)

	def dense_to_sparse(self, rgb, depth):
		"""
		Samples pixels with `num_samples`/#pixels probability in `depth`.
		Only pixels with a maximum depth of `max_depth` are considered.
		If no `max_depth` is given, samples in all pixels
		"""
		mask_keep = depth > 0
		if self.max_depth is not np.inf:
			mask_keep = np.bitwise_and(mask_keep, depth <= self.max_depth)
		n_keep = np.count_nonzero(mask_keep)
		if n_keep == 0:
			return mask_keep
		else:
			prob = float(self.num_samples) / n_keep
			return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)


class SimulatedStereo(DenseToSparse):
	name = "sim_stereo"

	def __init__(self, num_samples, max_depth=np.inf, dilate_kernel=3, dilate_iterations=1):
		DenseToSparse.__init__(self)
		self.num_samples = num_samples
		self.max_depth = max_depth
		self.dilate_kernel = dilate_kernel
		self.dilate_iterations = dilate_iterations

	def __repr__(self):
		return "%s{ns=%d,md=%f,dil=%d.%d}" % \
			   (self.name, self.num_samples, self.max_depth, self.dilate_kernel, self.dilate_iterations)

	# We do not use cv2.Canny, since that applies non max suppression
	# So we simply do
	# RGB to intensitities
	# Smooth with gaussian
	# Take simple sobel gradients
	# Threshold the edge gradient
	# Dilatate
	def dense_to_sparse(self, rgb, depth):
		gray = rgb2grayscale(rgb)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
		gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

		depth_mask = np.bitwise_and(depth != 0.0, depth <= self.max_depth)

		edge_fraction = float(self.num_samples) / np.size(depth)

		mag = cv2.magnitude(gx, gy)
		min_mag = np.percentile(mag[depth_mask], 100 * (1.0 - edge_fraction))
		mag_mask = mag >= min_mag

		if self.dilate_iterations >= 0:
			kernel = np.ones((self.dilate_kernel, self.dilate_kernel), dtype=np.uint8)
			cv2.dilate(mag_mask.astype(np.uint8), kernel, iterations=self.dilate_iterations)

		mask = np.bitwise_and(mag_mask, depth_mask)
		return mask
