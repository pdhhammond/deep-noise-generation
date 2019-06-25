"""
Routines for generating algorithmic noise.
"""

import numpy as np
import cv2

import time as t

def gradient_circle(radius):
	matrix = np.zeros([radius*2+1,radius*2+1], np.float32)

	radius_sqr = np.square(radius)
	diffs = np.arange(2*radius+1) - radius
	sqr_diffs = np.square(diffs)
	rows = diffs + radius
	cols = diffs + radius

	for r, row in enumerate(rows):
		if row < 0 or row >= matrix.shape[0]: continue
		for c, col in enumerate(cols):
			if col < 0 or col >= matrix.shape[1]: continue

			denom = (sqr_diffs[r] + sqr_diffs[c])
			if denom == 0:
				matrix[row, col] = 1
			elif radius_sqr / denom >= 1:
				matrix[row, col] = 1. - denom / radius_sqr
	return matrix

def noise_mask_for_region(depth, low, high, prob=.5):
	noise = np.random.choice(2, size=depth.shape, p=[prob, 1.0-prob]).astype(np.float32)
	noise[depth < low] = 1.
	noise[depth > high] = 1.
	return noise

def mask_out_random(img_stack, prob=.5):
	mask = np.random.choice(2, size=img_stack.shape, p=[prob, 1.0-prob])
	return (img_stack * mask).astype(img_stack.dtype)

def close(image, kernel, iterations):
	image = cv2.erode(image, kernel, iterations=iterations)
	return cv2.dilate(image, kernel, iterations=iterations)

def get_edges(image, threshold):
	edges = np.abs(cv2.Laplacian(image, cv2.CV_64F)).astype(np.uint8)
	edges[edges < threshold] = 0
	edges[edges >= threshold] = 255
	return edges

def generate_dropout_mask(image, med_depth):
	edges0 = get_edges((med_depth * 225).astype(np.uint8), 25)
	edges1 = get_edges(image, 25)
	edges0[edges1 > 0] = 255
	edges = mask_out_random(edges0, .8)

	return close(255 - edges, np.ones([3,3], np.uint8), 2)

def generate_depth_noise(mask, element):
	radius = int(element.shape[0]/2.)
	rows, cols = np.where(mask == 0)
	noise = np.zeros_like(mask)

	for i in range(rows.shape[0]):
		if rows[i] <= radius or rows[i] >= noise.shape[0] - radius: continue
		if cols[i] <= radius or cols[i] >= noise.shape[1] - radius: continue

		scale = np.random.choice((np.arange(10) + 1) / 10) * np.random.choice([-1,1])
		noise_circle = element * scale

		noise[rows[i]-radius-1 : rows[i]+radius, cols[i]-radius-1 : cols[i]+radius] += noise_circle

	return noise

# this is a terrible way to do this
c0 = gradient_circle(30) / 40.
c1 = gradient_circle(20) / 30.
c2 = gradient_circle(10) / 30.
c3 = gradient_circle(5) / 10.

def get_noise(image, depth):
	dropout_mask = generate_dropout_mask(image, depth)

	mask = noise_mask_for_region(depth, 0., 1., .0005)
	noise = generate_depth_noise(mask, c0)

	mask = noise_mask_for_region(depth, .3, 1., .001)
	noise = noise + generate_depth_noise(mask, c1)

	mask = noise_mask_for_region(depth, .4, 1., .01)
	noise = noise + generate_depth_noise(mask, c2)

	mask = noise_mask_for_region(depth, .8, 1., .01)
	noise = noise + generate_depth_noise(mask, c3)

	return dropout_mask, noise

def generate_naive_dropout_mask(med_depth, prob):
	ones = np.ones(med_depth.shape, np.uint8)
	return mask_out_random(ones, prob) * 255

def get_naive_noise(image, depth):
	dropout_mask = generate_naive_dropout_mask(depth, .5)

	mask = noise_mask_for_region(depth, 0., 1., .01)
	noise = generate_depth_noise(mask, c2)

	return dropout_mask, noise

def add_noise(image, depth, naive=False):
	noise_func = get_naive_noise if naive else get_noise
	dropout_mask, noise = noise_func(image, depth)

	out = np.clip(depth + noise, 0., 1.)
	out[dropout_mask == 0] = 0
	return out
