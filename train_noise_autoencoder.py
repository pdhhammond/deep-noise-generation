import numpy as np
import cv2
import os
import json
import argparse

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms

from Architectures.VAE import VariationalEncoderNet, VariationalDecoderNet

import DataLoaders.NoiseAutoencoder_DataLoader as dl

# height and width of images after normalization
img_size = 256
loss = nn.MSELoss()

# handle command line arguments
parser = argparse.ArgumentParser(description='Noise-Autoencoder')
parser.add_argument('data', help='name of data folder.')
parser.add_argument('fname', help='name of output folder.')
parser.add_argument('--epochs', '-e', default=1000, type=int, help=
	'number of epochs to train (default: 1000)')
parser.add_argument('--traincap', default=-1, type=int, help=
	'caps the number of training scenes (default: -1 for all possible samples)')
parser.add_argument('--valcap', default=-1, type=int, help=
	'caps the number of validation scenes (default: -1 for all possible' +
	' samples)')
parser.add_argument('--trainsample', default=-1, type=int, help='caps the' +
	' number of noise samples per training scene' +
	' (default: -1 for all possible samples)')
parser.add_argument('--valsample', default=-1, type=int, help='caps the' +
	' number of noise samples per validation scene (default: -1 for all' +
	' possible samples)')
parser.add_argument('--gpu', '-gpu', default=0, type=int, help='which gpu to' +
	' use (default: 0)')
parser.add_argument('--workers', '-w', default=10, type=int, help='number of' +
	' data loading workers (default: 10)')
args = parser.parse_args()


# helper methods for training
def get_error(generated_data, real_data):
	'''
	Computes a weighted MSE loss for the VAE network.

	Parameters
	----------
	generated_data : torch.autograd.Variable
		The generated output of the network.
	real_data : torch.autograd.Variable
		The target or real data that the network should have produced.

	Returns
	-------
	error : torch.autograd.Variable
		The computed network error.
	'''
	error0 = loss(generated_data[:, 0], real_data[:, 0])
	error1 = loss(generated_data[:, 1], real_data[:, 1]) * 1e2
	error = error0 + error1
	return error

def train_autoencoder(e_optim, d_optim, generated_data, real_data):
	'''
	Performs a single backpropagation step for training the VAE.

	Parameters
	----------
	e_optim : torch.optim
		The optimizer for the encoder network.
	d_optim : torch.optim
		The optimizer for the decoder network.
	generated_data : torch.autograd.Variable
		The generated output of the network.
	real_data : torch.autograd.Variable
		The target or real data that the network should have produced.

	Returns
	-------
	error : torch.autograd.Variable
		The computed network error.
	'''
	# Reset gradients
	e_optim.zero_grad()
	d_optim.zero_grad()

	# Calculate error and backpropagate
	error = get_error(generated_data, real_data)
	error.backward()

	# Update weights with gradients
	e_optim.step()
	d_optim.step()

	# Return error
	return error

# initialize image pre-processing pipeline
def depth_noise_data(data_root, descriptor, samples):
	'''
	Creates a DepthNoiseDataset with a pre-processing pipeline built in.

	Parameters
	----------
	data_root : str
		The path to the dataset.
	descriptior : list
		A dataset descriptor that indicates the indexes of the depth frames to
		sample from each folder. Should be organized:
		[(folder_name, [index_1, ...]), ...]
	samples : int
		The number of samples taken from each folder. Used when determining
		which proxy ground-truth images to load.

	Returns
	-------
	dl : DepthNoiseDataset
		The created DepthNoiseDataset with a built in pre-processing pipeline.
	'''
	compose = transforms.Compose([
		dl.Rescale((img_size, img_size)),
		dl.ToTensor(),
		dl.Normalize(),
		dl.SoftLabels()
		])
	return dl.DepthNoiseDataset(data_root, descriptor, samples, compose)

# helper method for generation
def noise(size, n_dims=100):
	'''
	Generates a 1-d vector of gaussian-sampled random values.

	Parameters
	----------
	size : int
		The number of vectors to generate.
	n_dims : int
		The dimensionality of the vectors to generate.

	Returns
	-------
	noise : torch.autograd.Variable
		The generated batch of random vectors.
	'''
	n = torch.randn(size, n_dims, 1, 1)
	n = Variable(nn.functional.normalize(n, p=2, dim=1).cuda())
	return n

def train(encoder, decoder, data_loader, e_optim, d_optim):
	'''
	Performs a single training epoch.

	Parameters
	----------
	encoder : torch.nn.Module
		The PyTorch encoder module.
	decoder: torch.nn.Module
		The PyTorch decoder module.
	data_loader : DataLoaders.VariationalEncoderNet
		The Dataset to use for training.
	e_optim : torch.nn.optim
		The optimizer for the encoder module.
	d_optim : torch.nn.optim
		The optimizer for the decoder module.

	Returns
	-------
	error : float
		The mean training error for the entire epoch.
	'''
	encoder.train() # switch to train mode
	decoder.train()
	errors = []

	# iterate over each batch
	for n_batch, batch in enumerate(data_loader):
		# get training batch
		image, depth = batch['image'], batch['depth']
		dropout, depth_noise = batch['dropout'], batch['noise']
		N = depth_noise.size(0)

		real_data = Variable(torch.cat([dropout, depth_noise], 1).cuda())
		real_labels = Variable(torch.cat([image, depth], 1).cuda())

		# generate fake data
		mu, sigma = encoder(real_data)
		gen_data = decoder(mu, sigma, real_labels)

		# perform the backpropagation step
		g_error = train_autoencoder(e_optim, d_optim, gen_data, real_data)
		errors += [g_error.data.cpu().numpy()]

	return float(np.mean(errors))

def validate(encoder, decoder, test_data_loader):
	'''
	Performs a single validation epoch.

	Parameters
	----------
	encoder : torch.nn.Module
		The PyTorch encoder module.
	decoder: torch.nn.Module
		The PyTorch decoder module.
	test_data_loader : DataLoaders.VariationalEncoderNet
		The Dataset to use for validation.

	Returns
	-------
	error : float
		The mean training error for the entire epoch.
	'''
	# test epoch
	encoder.eval()
	decoder.eval()
	errors = []

	# iterate over the entire validation set
	for n_batch, batch in enumerate(test_data_loader):
		# get training batch
		image, depth = batch['image'], batch['depth']
		dropout, depth_noise = batch['dropout'], batch['noise']
		N = depth_noise.size(0)

		real_data = Variable(torch.cat([dropout, depth_noise], 1).cuda())
		real_labels = Variable(torch.cat([image, depth], 1).cuda())

		# Test Autoencoder
		mu, sigma = encoder(real_data)
		gen_data = decoder(mu, sigma, real_labels)

		g_error = get_error(gen_data, real_data)
		errors += [g_error.data.cpu().numpy()]

	return float(np.mean(errors))

def get_visualizations(data_loader, batch_size=4):
	'''
	Precomputes variables for visualizations.

	Parameters
	----------
	data_loader : DataLoaders.VariationalEncoderNet
		The Dataset to use for training.
	batch_size : int
		The number of images to pre-compute.

	Returns
	-------
	vis_labels : torch.autograd.Variable
		The batch of ground-truth RGB-D images to use when generating
		visualizations.
	vis_dropout : numpy.ndarray
		A pre-computed horizontal stack of ground-truth dropout images.
	vis_depth_noise : numpy.ndarray
		A pre-computed horizontal stack of ground-truth residual noise images.
	'''
	vis_batch = next(iter(data_loader))
	dropout, depth_noise = vis_batch['dropout'][:4], vis_batch['noise'][:4]
	vis_data = torch.cat([dropout, depth_noise], 1)
	vis_data = Variable(vis_data.cuda())

	image, depth = vis_batch['image'][:4], vis_batch['depth'][:4]
	vis_labels = torch.cat([image, depth], 1)
	vis_labels = Variable(vis_labels.cuda())

	vis_dropout = np.squeeze(dropout.data.cpu().numpy())
	vis_dropout = np.hstack(vis_dropout)

	vis_depth_noise = np.squeeze(depth_noise.data.cpu().numpy())
	vis_depth_noise = np.hstack(vis_depth_noise)

	return vis_labels, vis_dropout, vis_depth_noise

def visualize(decoder, vis_labels, vis_dropout, vis_depth_noise,
				message="generated", delay=1):
	'''
	Displays a batch of generated noise compared with ground-truth.

	Parameters
	----------
	decoder: torch.nn.Module
		PyTorch decoder model.
	vis_labels : torch.autograd.Variable
		Batch of ground-truth RGB-D images to use to generate visualizations.
	vis_dropout : numpy.ndarray
		A pre-computed horizontal stack of ground-truth dropout images.
	vis_depth_noise : numpy.ndarray
		A pre-computed horizontal stack of ground-truth residual noise images.
	message : str
		Message to display on the CV2 window. Defaults to "generated".
	delay : int
		Number of milliseconds to display the window for. Defaults to 1,
		which will display the window without interrupting training until this
		function is called again.
	'''
	z = noise(vis_labels.shape[0])
	vis_images = decoder.generate(z, vis_labels)
	vis_images = vis_images.data.cpu().numpy()

	gen_dropout = vis_images[:, 0]
	gen_dropout[gen_dropout <= 0] = -1.
	gen_dropout[gen_dropout > 0] = 1.
	gen_dropout = np.hstack(gen_dropout)

	gen_depth_noise = np.hstack(vis_images[:, 1])

	vis_image = np.vstack([gen_dropout, gen_depth_noise, vis_dropout,
		vis_depth_noise]) / 2. + .5
	cv2.imshow(message, vis_image)
	cv2.waitKey(delay)


if __name__ == "__main__":
	# assign the GPU to use
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

	# create directory for output
	out_path = os.path.join("./autoencoder_results", args.fname)
	if not os.path.isdir(out_path):
		os.mkdir(out_path)

	# get the path to the training data
	data_path = args.data

	# initialize training and validation records
	best_error = float("inf")
	best_epoch = 0
	train_error = []
	test_error = []

	# total number of epochs to train
	num_epochs = args.epochs

	# number of training pairs per batch
	batch_size = 20


	# divide dataset into random test/training split
	# our formulation uses a random split of the training set for validation
	# when training the noise generator to prevent accidental overfit to the
	# actual validation set
	t_desc, v_desc = dl.partition_dataset(data_path, args.traincap, args.valcap,
		args.trainsample, args.valsample)

	# create Datasets for training and validation
	data = depth_noise_data(data_path, t_desc, args.trainsample)
	test = depth_noise_data(data_path, v_desc, args.valsample)

	# create DataLoaders with Datasets so that we can iterate over them
	data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
		shuffle=True, num_workers=args.workers)
	test_data_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
		shuffle=True, num_workers=args.workers)

	# get number of training and validation batches
	num_batches = len(data_loader)
	num_test_batches = len(test_data_loader)


	# create and initialize the VAE
	encoder = VariationalEncoderNet(100)
	encoder.weight_init(mean=.0, std=.02)
	encoder.cuda()

	decoder = VariationalDecoderNet(100)
	decoder.weight_init(mean=.0, std=.02)
	decoder.cuda()

	# create optimizers for training
	e_optim = optim.Adam(encoder.parameters(), lr=.0002, betas=(.5, .999))
	d_optim = optim.Adam(decoder.parameters(), lr=.0002, betas=(.5, .999))

	# prepare visualizations from validation set
	v_labels, v_dropout, v_depth_noise = get_visualizations(test_data_loader)


	# start training
	print("TRAINING FOR", num_epochs, "EPOCHS")

	# train for the specified number of epochs
	for epoch in range(num_epochs):
		# early terminating condition
		if len(test_error) > 50 and best_error not in test_error[-50:]:
			print("Hasn't improved in 50 epochs. Terminating early.")
			break

		# training epoch
		mean_error = train(encoder, decoder, data_loader, e_optim, d_optim)
		train_error += [mean_error]

		print("\ntrain epoch", "(" + str(epoch) + "/" + str(num_epochs) + ")",
			"g_error", train_error[-1])

		# validation epoch
		mean_error = validate(encoder, decoder, test_data_loader)
		test_error += [mean_error]

		print("test epoch", "(" + str(epoch) + "/" + str(num_epochs) + ")",
			"g_error", test_error[-1])

		# visualize model progress
		visualize(decoder, v_labels, v_dropout, v_depth_noise)

		# save weights if best error is improved
		if test_error[-1] < best_error:
			print("saved best")
			torch.save(decoder.state_dict(), os.path.join(out_path,
				"best_generator.json"))

			best_error = test_error[-1]
			best_epoch = epoch

		# record training log
		with open(os.path.join(out_path, "out_log.json"), 'w') as f:
			data = {'total_epochs':epoch,
					'best_epoch':best_epoch,
					'best_test_error':best_error,
					'train_record':train_error,
					'test_record':test_error}
			json.dump(data, f)