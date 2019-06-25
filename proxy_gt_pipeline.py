"""
Makes proxy ground-truth images for a directory of depth scenes. Does not apply
cross-bilateral filtering for hole filling. Use the separate Matlab script to
complete the pipeline.

Usage
-----
python pipeline.py [-h] [-c CUTOFF] [-w WORKERS] [-np no_prompt] data

Input
-----
CUTOFF : int
	Maximum depth distance. Clips all greater values to this number.
WORKERS : int
	Number of worker threads to spawn. Must be greater than 0.
no_prompt : flag
	Disables prompting before deleting files.
data : str
	Path to the folder containing all of the depth scenes.
"""

import threading
import argparse
import json
import sys
import os

import numpy as np
import util
import cv2

from time import time

# handle command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('data', help='name of data folder.')
parser.add_argument("-c", "--cutoff", type=float, default=10000,
	help="the cutoff distance for the raw depth files")
parser.add_argument("-w", "--workers", type=int, default=5,
	help="the number of worker threads")
parser.add_argument("-np", "--no_prompt", action="store_true",
	help="disables prompting before deleting files.")

def animate_until_input(frames):
	'''
	Prompts the user to delete moving frames out of the scene.
	'''
	i = 0
	response = False

	while True:
		i = i % frames.shape[0]

		cv2.imshow("Really delete %d frames? (Y/N)"
			% frames.shape[0], frames[i])
		key = cv2.waitKey(1) & 0xFF

		if key == ord("y"):
			response = True
		if key == ord("y") or key == ord("n"):
			break
		i += 1

	cv2.destroyAllWindows()
	return response

def delete_files(path_stub, indexes):
	for i in indexes:
		img_path = os.path.join(path_stub, "images/color_%04d.jpg" % i)
		raw_path = os.path.join(path_stub, "depths/raw_%04d.npy" % i)

		if os.path.exists(img_path):
			os.remove(img_path)
		if os.path.exists(raw_path):
			os.remove(raw_path)

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
	m_not_indices = np.where(collapsed <= 0.0002)[0]

	return m_indices, m_not_indices

def get_median_depth(raw_stack):
	'''
	Applies a median-stack filter function to a stack of raw depth images.
	'''
	if raw_stack.shape[0] == 2:
		med_depth = np.mean(raw_stack, axis=0)
		sum_depth = np.sum(raw_stack, axis=0)

		mask = np.logical_xor(raw_stack[0] == 0, raw_stack[1] == 0)
		med_depth[mask] = sum_depth[mask]
	else:
		med_depth = np.median(raw_stack, axis=0)
	return med_depth

def remove_speckles(med_dep, radius=20):
	'''
	Applies a circlular binary close function to remove speckles.
	'''
	missing_mask = np.zeros_like(med_dep).astype(np.uint8)
	missing_mask[med_dep == 0] = 255

	element = createCircularMask(radius, radius).astype(np.uint8)
	filtered_mask = close(missing_mask, element).astype(np.uint8)

	med_raw = np.copy(med_dep)
	med_raw[filtered_mask > 0] = 0
	med_dep[filtered_mask > 0] = 0
	return med_dep

def pipeline_worker(id, paths, cutoff, no_prompt):
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
		delete = no_prompt

		start = time()
		try:
			# check if folder exists
			if not os.path.isdir(cur_path): continue

			# load image stacks
			raw_path = os.path.join(cur_path, "depths")
			raw_stack, filenames = util.load_image_stack(raw_path)
			rgb_path = os.path.join(cur_path, "images")
			rgb_stack, filenames = util.load_image_stack(rgb_path, color_mode=1)

			# get median color images
			med_image = np.median(rgb_stack, axis=0)

			# delete any moving frames
			m_indices, m_not = get_motion_indices(rgb_stack, med_image)
			m_frames = rgb_stack[m_indices]

			# prompt user for input
			if not no_prompt and len(m_indices) > 0:
				delete = animate_until_input(m_frames)
				cv2.destroyAllWindows()
			# remove frames with motion and recalculate median color image
			if delete == True and m_frames.shape[0] > 0:
				delete_files(cur_path, m_indices)
				shape = m_indices.shape[0]
				print("%d:\tDeleted %d frames from %s" % (id, shape, cur_path))

				raw_stack = raw_stack[m_not]
				rgb_stack = rgb_stack[m_not]
				med_image = np.median(rgb_stack, axis=0)

			# compute proxy ground-truth depth image
			med_depth = get_median_depth(raw_stack)
			med_depth = remove_speckles(med_depth, 15)

			med_depth = (np.clip(med_depth, 0, cutoff-1) / float(cutoff))
			med_depth = (med_depth * 255).astype(np.uint8)

			# write proxy ground-truth images
			rgb_path = os.path.join(cur_path, "med_image.png")
			dep_path = os.path.join(cur_path, "med_depth.png")

			cv2.imwrite(rgb_path, med_image)
			cv2.imwrite(dep_path, med_depth)

			# report progress
			percent_complete = int(float((j+1.)/len(paths)) * 100)
			print("%d:\t%d/%d\t%d%% complete" % (id, j + 1, len(paths),
				percent_complete))
			print(time() - start)
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
	no_prompt = args.no_prompt

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
			args=(i, divisions[i], cutoff, no_prompt))
		threads.append(t)
		t.start()