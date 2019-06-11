"""
Makes median-stack images, dropout images, and residual depth images for a
directory of depth scenes.

Usage
-----
python pipeline.py [-h] [-c CUTOFF] [-w WORKERS] data

Input
-----
CUTOFF : int
	Maximum depth distance. Clips all greater values to this number.
WORKERS : int
	Number of worker threads to spawn. Must be greater than 0.
data : str
	Path to the folder containing all of the depth scenes.
"""

import threading
import argparse
import json
import sys
import os

import numpy as np
import cv2


# handle command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('data', help='name of data folder.')
parser.add_argument("-c", "--cutoff", type=float, default=10000,
	help="the cutoff distance for the raw depth files")
parser.add_argument("-w", "--workers", type=int, default=5,
	help="the number of worker threads")


def get_depth_image(raw, cutoff=10000):
	'''
	Converts a raw depth image into a ready-to-display uint8 image.

	Parameters
	----------
	raw : numpy.ndarray
		Raw depth image with shape [height, width] and type float.
	cutoff : int
		Distance to use for clipping. (Default: 10000, or 10 meters.)

	Returns
	-------
	depth_image : numpy.ndarray
		Clipped depth image scaled between 0 and 255. Shape [height, width],
		type uint8.
	'''
	return (np.clip(raw / cutoff, 0., 1.) * 255.).astype(np.uint8)

def divide_work(paths, num_workers):
	'''
	Naively divides the list of data paths between the allocated number of
	worker threads.

	Parameters
	----------
	paths : [str, ...]
		Complete list of data paths that need processing.
	num_workers : int
		The number of worker threads being initialized.

	Returns
	-------
	divisions : [[str, ...], ...]
		List of sub-divided path lists, one for each worker thread.
	'''
	divisions = []
	chunk_size = int(len(paths) / num_workers)

	# the workload division could be more efficient
	for i in range(num_workers):
		start_ind = i * chunk_size
		if i == num_workers-1:
			divisions += [paths[start_ind : ]]
		else:
			divisions += [paths[start_ind : start_ind + chunk_size]]
	return divisions

def get_motion_indices(frame_stack, med_image, threshold=102):
	'''
	Returns a list of indexes marking each frame that deviates from the median
	image by more than a threshold.

	Parameters
	----------
	frame_stack : numpy.ndarray
		The stack of RGB frames to check for motion. In format
		[frames, height, width, channels] with dtype numpy.uint8.
	med_image : numpy.ndarray (np.uint8)
		The result of taking the median over the stack of frames. With shape
		[height, width, channels] with dtype numpy.uint8.
	threshold : int
		Threshold for motion. Each frame with a mean absolute difference from
		the median image above this threshold will be flagged for motion.
		Defaults to 102, which was appropriate for most of our dataset.

	Returns
	-------
	m_indices : numpy.ndarray
		Indexes of frames with motion. In format [int, ...].
	'''
	diffs = frame_stack.astype(np.float32) - med_image.astype(np.float32)
	abs_diffs = np.abs(diffs)

	# if the images are color, take the average over each channel
	if len(frame_stack.shape) > 3:
		abs_diffs = np.mean(abs_diffs, axis=-1)

	# mark all pixels with a greater difference than the threshold
	diff_mask = np.zeros_like(abs_diffs)
	diff_mask[abs_diffs >= threshold] = 1.

	# gets the average number of pixels in the frame that are in motion
	new_shape = [diff_mask.shape[0], np.product(diff_mask.shape[1:])]
	collapsed = np.reshape(diff_mask, new_shape)
	collapsed = np.mean(collapsed, axis=-1)

	# marks each frame as moving if a certain number of pixels are in motion
	m_indices = np.where(collapsed > 0.0002)[0]

	return m_indices

def get_image_stacks(root_path):
	'''
	Loads depth and color image stacks from the given directory.

	Parameters
	----------
	root_path : str
		The path to the scene directory. Must contain subfolders titled depths
		and images.

	Returns
	-------
	raw_stack : numpy.ndarray
		Stack of raw depth frames with shape [num_images, height, width] and
		type numpy.float32.
	rgb_stack : numpy.ndarray
		Stack of color frames with shape [num_images, height, width, channels]
		and type numpy.uint8.
	'''
	raw_path = os.path.join(root_path, "depths")
	rgb_path = os.path.join(root_path, "images")
	raw_stack = []
	rgb_stack = []

	rfiles = next(os.walk(raw_path))[2]
	cfiles = next(os.walk(rgb_path))[2]

	# gather the raw depth images
	for i in range(len(rfiles)):
		raw = np.load(os.path.join(raw_path, rfiles[i]))
		rgb = cv2.imread(os.path.join(rgb_path, cfiles[i]), 1)
		raw_stack += [raw]
		rgb_stack += [rgb]

	raw_stack = np.array(raw_stack).astype(np.float32)
	rgb_stack = np.array(rgb_stack)
	return raw_stack, rgb_stack

def get_median_depth(raw_stack, cutoff):
	'''
	Makes a median-stack depth image from a stack of raw depth frames. Ignores
	dropout when calculating the median. Output image will only contain dropout
	if no frame contained a valid non-zero value at a given pixel.

	Parameters
	----------
	raw_stack : numpy.ndarray
		Stack of raw depth frames of shape [num_frames, height, width] of type
		float.
	cutoff : int
		Maximum depth distance. Clips all greater values to this number.

	Returns
	-------
	med_raw : numpy.ndarray
		1-channel median depth image taken from the image stack. Of shape
		[height, width] with the same type as raw_stack.
	'''
	# get median depth image
	med_raw = np.median(raw_stack, axis=0)
	med_dep = get_depth_image(med_raw, cutoff)

	# mark 0 values with nan
	raw_stack = np.copy(raw_stack)
	raw_stack[raw_stack == 0] = np.nan

	# take the median ignoring the nans (0s)
	# this helps to prevent modeling dropout regions as depth noise
	med_raw = np.nanmedian(raw_stack, axis=0)
	med_raw[np.isnan(med_raw)] = 0

	return med_raw

def get_residuals(raw_stack, med_raw, dropout_masks, cutoff):
	'''
	Gets residual noise images by subtracting a median depth image from raw
	depth frames. Dropout regions are ignored.

	Parameters
	----------
	raw_stack : numpy.ndarray
		Stack of raw depth frames of shape [num_frames, height, width] of type
		float.
	med_raw : numpy.ndarray
		1-channel median depth image taken from the image stack. Of shape
		[height, width] with the same type as raw_stack.
	dropout_masks : numpy.ndarray
		Stack of binary dropout masks of shape [num_frames, height, width] with
		type int.
	cutoff : int
		Maximum depth distance. Clips all greater values to this number.

	Returns
	-------
	noise_stack : numpy.ndarray
		Stack of residual depth frames computed from the raw depth frames and
		the median-stack depth image. With shape [num_frames, height, width] and
		type numpy.uint8.
	'''
	noise_stack = np.clip((raw_stack - med_raw) / cutoff, -1., 1.)
	noise_stack = noise_stack / 2. + .5

	noise_stack[dropout_masks == 0] = .5
	noise_stack = (noise_stack * 255.).astype(np.uint8)

	return (noise_stack * 255.).astype(np.uint8)

def save_image_stack(root_path, folder_name, image_stack):
	'''
	Saves a stack of images as individual png files. Saves all frames under the
	directory root_path/folder_name.

	Parameters
	----------
	root_path : str
		Path to the root directory to write the output folder.
	folder_name : str
		The name of the output directory.
	image_stack : []
		List-like object containing each image frame along the first dimension.
		All images must be of type uint8.
	'''
	dir = os.path.join(root_path, folder_name)
	if not os.path.exists(dir):
		os.makedirs(dir)
	for i in range(image_stack.shape[0]):
		filename = os.path.join(dir, "%s_%04d.png") % (folder_name, i)
		cv2.imwrite(filename, image_stack[i])

def pipeline_worker(id, paths, cutoff):
	'''
	Generates median depth images, dropout frames, and residual depth frames
	for a list of depth scenes.

	Parameters
	----------
	id : int
		Identifying number of the thread. Used for reporting to the console.
	paths : []
		List of paths to each depth scene the thread should process.
	cutoff : int
		Maximum depth distance. Clips all greater values to this number.
	'''
	for j, cur_path in enumerate(paths):
		try:
			# check if folder exists
			if not os.path.isdir(cur_path): continue
			# load image stacks
			raw_stack, rgb_stack = get_image_stacks(cur_path)

			# get median images
			med_image = np.median(rgb_stack, axis=0)
			med_raw = get_median_depth(raw_stack, cutoff)

			# get dropout masks
			drop_mask = np.ones_like(raw_stack).astype(np.uint8)  * 255.
			drop_mask[raw_stack == 0] = 0

			# get noise residuals
			noise_stack = get_residuals(raw_stack, med_raw, drop_mask, cutoff)

			# do not write out frames that are moving
			m_indices = get_motion_indices(rgb_stack, np.copy(med_image))
			drop_mask = np.delete(drop_mask, m_indices, axis=0)
			noise_stack = np.delete(noise_stack, m_indices, axis=0)

			# save indexes of moving fames for visualizations
			path_out = os.path.join(cur_path, "moving_frames.json")
			with open(path_out, 'w') as outfile:
				json.dump(m_indices.tolist(), outfile)

			# save dropout and noise stacks
			save_image_stack(cur_path, "dropout", drop_mask)
			save_image_stack(cur_path, "noise", noise_stack)

			# save the ground truth images
			np.save(os.path.join(cur_path, "med_raw.npy"),
				med_raw.astype(np.uint16))
			cv2.imwrite(os.path.join(cur_path, "med_depth.png"),
				get_depth_image(med_raw, cutoff))
			cv2.imwrite(os.path.join(cur_path, "med_image.png"), med_image)

			# report progress
			percent_complete = int(float((j+1.)/len(paths)) * 100)
			print("%d:\t%d/%d\t%d%% complete" % (id, j + 1, len(paths),
				percent_complete))
		except Exception as e:
			print("%d:\tERROR! %s" % (id, e))
			print("\tProblem processing %s\n\tAttempting to continue..."
				% cur_path)
			continue
	print("%d:\tThe dark deed is done." % id)
	return


if __name__ == "__main__":
	args = parser.parse_args()
	root_path = args.data
	cutoff = args.cutoff
	workers = args.workers

	# get the list of folders in the dataset
	paths = next(os.walk(root_path))[1]
	paths = [os.path.join(root_path, path) for path in paths]
	paths = paths[:5]
	paths.sort()

	# partition out the work
	if len(paths) < workers:
		workers = len(paths)
	divisions = divide_work(paths, workers)

	# create worker threads
	threads = []
	for i in range(workers):
		t = threading.Thread(target=pipeline_worker,
			args=(i, divisions[i], cutoff))
		threads.append(t)
		t.start()
