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
	d_optimizer = optim.Adam(encoder.parameters(), lr=.0002, betas=(.5, .999))
	g_optimizer = optim.Adam(decoder.parameters(), lr=.0002, betas=(.5, .999))


	# prepare visualizations from validation set
	test_batch = next(iter(data_loader))
	dropout, depth_noise = test_batch['dropout'][:4], test_batch['noise'][:4]
	test_data = torch.cat([dropout, depth_noise], 1)
	test_data = Variable(test_data.cuda())

	# pre-compute ground-truth visualizations
	v_image, v_depth = test_batch['image'][:4], test_batch['depth'][:4]
	test_labels = torch.cat([v_image, v_depth], 1)
	test_labels = Variable(test_labels.cuda())

	test_dropout = np.squeeze(dropout.data.cpu().numpy())
	test_dropout = np.hstack(test_dropout)

	test_depth_noise = np.squeeze(depth_noise.data.cpu().numpy())
	test_depth_noise = np.hstack(test_depth_noise)


	# start training
	print("TRAINING FOR", num_epochs, "EPOCHS")

	# train for the specified number of epochs
	for epoch in range(num_epochs):
		# early terminating condition
		if len(test_error) > 50 and best_error not in test_error[-50:]:
			print("Hasn't improved in 50 epochs. Terminating early.")
			break

		encoder.train()
		decoder.train()
		# keep track of error over this specific epoch
		errors = []

		# iterate over each batch
		for n_batch, batch in enumerate(data_loader):
			# get training batch
			image, depth = batch['image'], batch['depth']
			dropout, depth_noise = batch['dropout'], batch['noise']
			N = depth_noise.size(0)

			real_data = Variable(torch.cat([dropout, depth_noise], 1).cuda())
			real_labels = Variable(torch.cat([image, depth], 1).cuda())

			# 1. train Autoencoder
			# generate fake data
			mu, sigma = encoder(real_data)
			gen_data = decoder(mu, sigma, real_labels)

			# perform the backpropagation step
			g_error = train_autoencoder(d_optimizer, g_optimizer, gen_data,
				real_data)
			errors += [g_error.data.cpu().numpy()]

			# display Progress every few batches
			if n_batch % 10 == 0:
				z = noise(4)
				test_images = decoder.generate(z, test_labels)
				test_images = test_images.data.cpu().numpy()

				gen_dropout = test_images[:, 0]
				gen_dropout[gen_dropout <= 0] = -1.
				gen_dropout[gen_dropout > 0] = 1.
				gen_dropout = np.hstack(gen_dropout)

				gen_depth_noise = np.hstack(test_images[:, 1])

				test_image = np.vstack([gen_dropout, gen_depth_noise,
					test_dropout, test_depth_noise]) / 2. + .5
				cv2.imshow("generated", test_image)
				cv2.waitKey(1)

		# save training error and display progress
		train_error += [float(np.mean(errors))]
		print("train epoch", "(" + str(epoch) + "/" + str(num_epochs) + ")",
			"g_error", train_error[-1])

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

		# record mean validation error for this epoch
		test_error += [float(np.mean(errors))]
		print("test epoch", "(" + str(epoch) + "/" + str(num_epochs) + ")",
			"g_error", test_error[-1])
		print(out_path)

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

		# print spacer
		print()