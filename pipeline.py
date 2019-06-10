import threading
import argparse
import json
import sys
import os

import numpy as np
import cv2

def get_depth_image(raw, cutoff=10000):
	return (np.clip(raw / cutoff, 0., 1.) * 255.).astype(np.uint8)

def divide_work(paths, num_workers):
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
		[frames, width, height, channels] with dtype numpy.uint8.
	med_image : numpy.ndarray (np.uint8)
		The result of taking the median over the stack of frames. In format
		[width, height, channels] with dtype numpy.uint8.
	threshold : int
		Threshold for motion. Each frame with a mean absolute difference from
		the median image above this threshold will be flagged for motion.
		Defaults to 102, which was appropriate for most of our dataset.

	Returns
	-------
	m_indices : numpy.ndarray
		Indexes of frames with motion. In format [int, ...].
	'''
	abs_diffs = np.abs(frame_stack.astype(np.float32) - med_image.astype(np.float32))

	# if the images are color, take the average over each channel
	if len(frame_stack.shape) > 3:
		abs_diffs = np.mean(abs_diffs, axis=-1)

	# mark all pixels with a greater difference than the threshold
	diff_mask = np.zeros_like(abs_diffs)
	diff_mask[abs_diffs >= threshold] = 1.

	# gets the average number of pixels in the frame that are in motion
	collapsed = np.reshape(diff_mask, [diff_mask.shape[0], np.product(diff_mask.shape[1:])])
	collapsed = np.mean(collapsed, axis=-1)

	# marks each frame as moving if a certain number of pixels are in motion
	m_indices = np.where(collapsed > 0.0002)[0]

	return m_indices

def worker(id, paths, cutoff):
	print(str(id) + ":\tworking on", paths[0], "to", paths[-1])
	for j, cur_path in enumerate(paths):
		# try:
			# check if folder exists
		if not os.path.isdir(cur_path): continue

		raw_path = os.path.join(cur_path, "depths")
		rgb_path = os.path.join(cur_path, "images")
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

		# take the color median
		rgb_stack = np.array(rgb_stack)
		med_image = np.median(rgb_stack, axis=0)

		# stack the depth images
		raw_stack = np.array(raw_stack).astype(np.float32)

		# get dropout masks
		dropout_mask = np.ones_like(raw_stack).astype(np.uint8)  * 255.
		dropout_mask[raw_stack == 0] = 0

		# get median depth image
		med_raw = np.median(raw_stack, axis=0)
		med_dep = get_depth_image(med_raw, cutoff)

		# mark 0 values with nan
		raw_stack[raw_stack == 0] = np.nan

		# take the median ignoring the nans (0s)
		# this helps to prevent modeling dropout regions as depth noise
		med_raw = np.nanmedian(raw_stack, axis=0)
		med_raw[np.isnan(med_raw)] = 0
		med_dep = get_depth_image(med_raw, cutoff)

		# get rid of the nans in the raw stack
		raw_stack[np.isnan(raw_stack)] = 0

		# get noise residuals
		noise_stack = np.clip((raw_stack - med_raw) / cutoff, -1., 1.) / 2. + .5
		noise_stack[dropout_mask == 0] = .5
		noise_stack = (noise_stack * 255.).astype(np.uint8)

		# mark frames with motion
		m_indices = get_motion_indices(rgb_stack, np.copy(med_image))

		# do not write out frames that are moving
		dropout_mask = np.delete(dropout_mask, m_indices, axis=0)
		noise_stack = np.delete(noise_stack, m_indices, axis=0)

		# save moving fames for visualizations
		with open(os.path.join(cur_path, "moving_frames.json"), 'w') as outfile:
			json.dump(m_indices.tolist(), outfile)

		# save dropout masks
		dropout_dir = os.path.join(cur_path, "dropout")
		if not os.path.exists(dropout_dir):
			os.makedirs(dropout_dir)
		for i in range(dropout_mask.shape[0]):
			filename = os.path.join(dropout_dir, "dropout_%04d.png") % i
			cv2.imwrite(filename, dropout_mask[i])

		# save noise images
		noise_dir = os.path.join(cur_path, "noise")
		if not os.path.exists(noise_dir):
			os.makedirs(noise_dir)
		for i in range(noise_stack.shape[0]):
			filename = os.path.join(noise_dir, "noise_%04d.png") % i
			cv2.imwrite(filename, noise_stack[i])

		# save the ground truth images
		np.save(os.path.join(cur_path, "med_raw.npy"), med_raw.astype(np.uint16))
		cv2.imwrite(os.path.join(cur_path, "med_depth.png"), med_dep)
		cv2.imwrite(os.path.join(cur_path, "med_image.png"), med_image)

		# report progress
		percent_complete = int(float((j+1.)/len(paths)) * 100)
		print(str(id) + ":\t", str(j + 1) + "/" + str(len(paths)), "\t" + str(percent_complete) + "% complete\t")#, m_indices.shape[0], "frames w/ motion")
		# except:
		# 	print(str(id) + ":\tERROR! Problem processing", cur_path, "Attempting to continue.")
		# 	continue
	print(str(id) + ":\tThe dark deed is done.")
	return


parser = argparse.ArgumentParser()
parser.add_argument("integers", metavar="N", type=int, nargs="*", default=[1], help="start and stop indices")
parser.add_argument("-c", "--cutoff", type=float, default=10000, help="the cutoff distance for the raw depth files")
parser.add_argument("-w", "--workers", type=int, default=5, help="the number of worker threads")

if __name__ == "__main__":
	args = parser.parse_args()
	start = args.integers[0]
	end = (args.integers[-1] if len(args.integers) > 1 else start) + 1
	cutoff = args.cutoff
	workers = args.workers

	# paths = ["F:/d435_val/depths_vid_%04d" % i for i in range(start, end)]
	# paths = ["./D415_MovementExample/depths_vid_0401"]
	paths = ["C:\\Users\\pdhha\\OneDrive\\Documents\\DepthPictureTaker\\d415\\depths_vid_0000"]
	if len(paths) < workers:
		workers = len(paths)
	divisions = divide_work(paths, workers)

	threads = []
	for i in range(workers):
		t = threading.Thread(target=worker, args=(i, divisions[i], cutoff))
		threads.append(t)
		t.start()
