"""
Trains a depth-completor network. Technically set up to work on the NYU Depth V2
and Kitti Cityscapes datasets, but may not work in this versions. Specifically
set up to work on the D415 and D435 datasets.

From https://github.com/fangchangma/sparse-to-dense.pytorch
Modified to work with our data augmentation method and dense noisy RGB-D images

I don't list all of the input parameters. You can type in -h to see those. If I
don't list it here, it was left to its default setting in the experiments.

Usage
-----
usage: train_depth_completer.py [-h] [--arch ARCH] [--data DATA]
                                [--modality MODALITY] [-s N] [--max-depth D]
                                [--sparsifier SPARSIFIER]
                                [--sparsemodel SPARSEMODEL]
                                [--numaugmented NUMAUGMENTED] [--simoffline]
                                [--decoder DECODER] [-j N] [--epochs N]
                                [-c LOSS] [-b BATCH_SIZE] [--lr LR]
                                [--momentum M] [--weight-decay W]
                                [--print-freq N] [--resume PATH] [-e EVALUATE]
                                [--gpu N] [--fname FNAME] [--no-pretrain]
                                [--samplecap SAMPLECAP] [--scenecap SCENECAP]

Input
-----
DATA : str
 	dataset: gt_nyudepthv2 | nyudepthv2 | kitti | full pathname to data
	directory. (default: nyudepthv2)
MODALITY : str
	modality: rgb | rgbd | d (default: rgbd)
	Needs to be set to rgbd to work with our augmented training and our D415 and
	D435 datasets.
SPARSIFIER : str
	sparsifier: uar | sim_stereo | sim_camera | algorithmic | None
	(default: sim_camera) sim_camera is our noise-generating augmentation method.
SPARSEMODEL : str
	Pathname to the sparsifier model weights. (default: none)
NUMAUGMENTED : int
	The number of augmented frames per scene (default: 0 for no augmented data)
SIMOFFLINE : flag
	Use static augmented examples as if they had been pre-generated.
DECODER : str
	decoder: deconv2 | deconv3 | deconv5 | deconv8 | upconv | upproj
	(default: deconv5)
WORKERS : int
	Number of data loading workers (default: 10)
EPOCHS : int
	Number of total epochs to run (default: 15)
BATCH_SIZE : int
	Mini-batch size (default: 8)
GPU : int
	Which gpu to use (default: 0)
FNAME : str
	Name of output folder. If none, is automatically generated.
SAMPLECAP : int
	The number of noisy depth frames to sample per scene (default: -1 for all)
SCENECAP : int
	The number of scenes to sample (default: -1 for all)
"""

import os
import time
import csv
import numpy as np

import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

from models import ResNet
from metrics import AverageMeter, Result
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo, SimulatedCameraNoise, AlgorithmicNoise
import criteria
import depth_completer_utils as utils

args = utils.parse_command()
print(args)

# should remove this line if you want to do multi-gpu training
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
				'delta1', 'delta2', 'delta3',
				'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()

def create_data_loaders(args):
	# Data loading code
	print("=> creating data loaders ...")
	traindir = os.path.join('data', args.data, 'train')
	valdir = os.path.join('data', args.data, 'val')

	train_loader = None
	val_loader = None

	# sparsifier is a class for generating random sparse depth input from the ground truth
	sparsifier = None
	max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf

	if args.sparsifier == SimulatedCameraNoise.name:
		sparsifier = SimulatedCameraNoise(args.sparsemodel, 100)
	elif args.sparsifier == AlgorithmicNoise.name:
		sparsifier = AlgorithmicNoise()
	elif args.sparsifier == UniformSampling.name:
		sparsifier = UniformSampling(num_samples=args.num_samples,
		max_depth=max_depth)
	elif args.sparsifier == SimulatedStereo.name:
		sparsifier = SimulatedStereo(num_samples=args.num_samples,
		max_depth=max_depth)

	if args.data == 'nyudepthv2' or args.data == 'gt_nyudepthv2':
		from dataloaders.DepthNoise_DataLoader import DepthNoiseDataset
		if not args.evaluate:
			train_dataset = DepthNoiseDataset(traindir, type='train',
				sparsifier=sparsifier, num_augmented=args.numaugmented,
				sample_cap=args.samplecap, scene_cap=args.scenecap,
				sim_offline=args.simoffline)

		val_dataset = DepthNoiseDataset(valdir, type='val',
			sparsifier=None, num_augmented=0, sample_cap=1, scene_cap=-1)

	elif args.data == 'kitti':
		from dataloaders.kitti_dataloader import KITTIDataset
		if not args.evaluate:
			train_dataset = KITTIDataset(traindir, type='train',
				modality=args.modality, sparsifier=sparsifier)
		val_dataset = KITTIDataset(valdir, type='val',
			modality=args.modality, sparsifier=sparsifier)

	else:
		print("Uknown dataset. Attempting to load as D415 or D435.")
		from dataloaders.DepthNoise_DataLoader import DepthNoiseDataset
		if not args.evaluate:
			train_dataset = DepthNoiseDataset(traindir, type='train',
				sparsifier=sparsifier, num_augmented=args.numaugmented,
				sample_cap=args.samplecap, scene_cap=args.scenecap,
				sim_offline=args.simoffline)

		val_dataset = DepthNoiseDataset(valdir, type='val',
			sparsifier=None, num_augmented=0, sample_cap=1, scene_cap=-1)

	# set batch size to be 1 for validation
	val_loader = torch.utils.data.DataLoader(val_dataset,
		batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

	# put construction of train loader here, for those who are interested in testing only
	if not args.evaluate:
		train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=args.batch_size, shuffle=True,
			num_workers=args.workers, pin_memory=True, sampler=None,
			worker_init_fn=lambda work_id:np.random.seed(work_id))
			# worker_init_fn ensures different sampling patterns for each data loading thread

	print("=> data loaders created.")
	return train_loader, val_loader

def main():
	global args, best_result, output_directory, train_csv, test_csv

	# evaluation mode
	start_epoch = 0
	if args.evaluate:
		assert os.path.isfile(args.evaluate), \
		"=> no best model found at '{}'".format(args.evaluate)
		print("=> loading best model '{}'".format(args.evaluate))
		checkpoint = torch.load(args.evaluate)
		output_directory = os.path.dirname(args.evaluate)
		args = checkpoint['args']
		start_epoch = checkpoint['epoch'] + 1
		best_result = checkpoint['best_result']
		model = checkpoint['model']
		print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
		_, val_loader = create_data_loaders(args)
		args.evaluate = True
		avg, img_merge = validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
		print(img_merge.dtype, np.min(img_merge), np.max(img_merge))

		img_merge[:, 304:] = img_merge[:, 304:, ::-1]
		cv2.imwrite("BUNNIES.png", img_merge)
		return

	# optionally resume from a checkpoint
	elif args.resume:
		assert os.path.isfile(args.resume), \
			"=> no checkpoint found at '{}'".format(args.resume)
		print("=> loading checkpoint '{}'".format(args.resume))
		checkpoint = torch.load(args.resume)
		args = checkpoint['args']
		start_epoch = checkpoint['epoch'] + 1
		best_result = checkpoint['best_result']
		model = checkpoint['model']
		optimizer = checkpoint['optimizer']
		output_directory = os.path.dirname(os.path.abspath(args.resume))
		print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
		train_loader, val_loader = create_data_loaders(args)
		args.resume = True

	# create new model
	else:
		train_loader, val_loader = create_data_loaders(args)
		print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
		in_channels = len(args.modality)
		if args.arch == 'resnet50':
			model = ResNet(layers=50, decoder=args.decoder, output_size=train_loader.dataset.output_size,
				in_channels=in_channels, pretrained=args.pretrained)
		elif args.arch == 'resnet18':
			model = ResNet(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
				in_channels=in_channels, pretrained=args.pretrained)
		print("=> model created.")
		optimizer = torch.optim.SGD(model.parameters(), args.lr, \
			momentum=args.momentum, weight_decay=args.weight_decay)

		# model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
		model = model.cuda()

	# define loss function (criterion) and optimizer
	if args.criterion == 'l2':
		criterion = criteria.MaskedMSELoss().cuda()
	elif args.criterion == 'l1':
		criterion = criteria.MaskedL1Loss().cuda()

	# create results folder, if not already exists
	output_directory = utils.get_output_directory(args) if args.fname is None else os.path.join("./results", args.fname)
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
	train_csv = os.path.join(output_directory, 'train.csv')
	test_csv = os.path.join(output_directory, 'test.csv')
	best_txt = os.path.join(output_directory, 'best.txt')

	# create new csv files with only header
	if not args.resume:
		with open(train_csv, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()
		with open(test_csv, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()

	for epoch in range(start_epoch, args.epochs):
		utils.adjust_learning_rate(optimizer, epoch, args.lr)
		train(train_loader, model, criterion, optimizer, epoch) # train for one epoch
		result, img_merge = validate(val_loader, model, epoch) # evaluate on validation set

		# remember best rmse and save checkpoint
		is_best = result.rmse < best_result.rmse
		if is_best:
			best_result = result
			with open(best_txt, 'w') as txtfile:
				txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
					format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
			if img_merge is not None:
				img_filename = output_directory + '/comparison_best.png'
				utils.save_image(img_merge, img_filename)
			print("Saved best")

		utils.save_checkpoint({
			'args': args,
			'epoch': epoch,
			'arch': args.arch,
			'model': model,
			'best_result': best_result,
			'optimizer' : optimizer,
		}, is_best, epoch, output_directory)


def train(train_loader, model, criterion, optimizer, epoch):
	average_meter = AverageMeter()
	model.train() # switch to train mode
	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		input, target = input.cuda(), target.cuda()
		torch.cuda.synchronize()
		data_time = time.time() - end

		# compute pred
		end = time.time()
		pred = model(input)
		loss = criterion(pred, target)
		optimizer.zero_grad()
		loss.backward() # compute gradient and do SGD step
		optimizer.step()
		torch.cuda.synchronize()
		gpu_time = time.time() - end

		depth_in = np.hstack(input.data.cpu().numpy()[:4, 3] / 10.)
		depth_in = cv2.applyColorMap((depth_in * 255).astype(np.uint8), cv2.COLORMAP_HOT)

		tgt_out = np.hstack(np.squeeze(target[:4].data.cpu().numpy())) / 10.
		tgt_out = cv2.applyColorMap((tgt_out * 255).astype(np.uint8), cv2.COLORMAP_HOT)

		out = np.hstack(np.squeeze(pred[:4].data.cpu().numpy()))
		out = np.clip(out / 10., 0., 1.)
		out = cv2.applyColorMap((out * 255).astype(np.uint8), cv2.COLORMAP_HOT)

		if i % 20 == 0:
			cv2.imshow("Training Results", np.vstack([depth_in, tgt_out, out]))
			cv2.waitKey(1)

		# measure accuracy and record loss
		result = Result()
		result.evaluate(pred.data, target.data)
		average_meter.update(result, gpu_time, data_time, input.size(0))
		end = time.time()

		if (i + 1) % args.print_freq == 0:
			print('=> output: {}'.format(output_directory))
			print('Train Epoch: {0} [{1}/{2}]\t'
				  't_Data={data_time:.3f}({average.data_time:.3f}) '
				  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
				  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
				  'MAE={result.mae:.2f}({average.mae:.2f}) '
				  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
				  'REL={result.absrel:.3f}({average.absrel:.3f}) '
				  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
				  epoch, i+1, len(train_loader), data_time=data_time,
				  gpu_time=gpu_time, result=result, average=average_meter.average()))

	avg = average_meter.average()
	with open(train_csv, 'a') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
			'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
			'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


def validate(val_loader, model, epoch, write_to_file=True):
	average_meter = AverageMeter()
	model.eval() # switch to evaluate mode
	end = time.time()
	for i, (input, target) in enumerate(val_loader):
		input, target = input.cuda(), target.cuda()
		torch.cuda.synchronize()
		data_time = time.time() - end

		# compute output
		end = time.time()
		with torch.no_grad():
			pred = model(input)
		torch.cuda.synchronize()
		gpu_time = time.time() - end

		# measure accuracy and record loss
		result = Result()
		result.evaluate(pred.data, target.data)
		average_meter.update(result, gpu_time, data_time, input.size(0))
		end = time.time()

		# save 8 images for visualization
		skip = 50
		if args.modality == 'd':
			img_merge = None
		else:
			if args.modality == 'rgb':
				rgb = input
			elif args.modality == 'rgbd':
				rgb = input[:,:3,:,:]
				depth = input[:,3:,:,:]

			if i == 0:
				if args.modality == 'rgbd':
					img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
				else:
					img_merge = utils.merge_into_row(rgb, target, pred)
			elif (i < 8*skip) and (i % skip == 0):
				if args.modality == 'rgbd':
					row = utils.merge_into_row_with_gt(rgb, depth, target, pred)
				else:
					row = utils.merge_into_row(rgb, target, pred)
				img_merge = utils.add_row(img_merge, row)
			elif i == 8*skip:
				filename = output_directory + '/comparison_' + str(epoch) + '.png'
				utils.save_image(img_merge, filename)

		if (i+1) % args.print_freq == 0:
			print('Test: [{0}/{1}]\t'
				  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
				  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
				  'MAE={result.mae:.2f}({average.mae:.2f}) '
				  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
				  'REL={result.absrel:.3f}({average.absrel:.3f}) '
				  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
				   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

	avg = average_meter.average()

	print('\n*\n'
		'RMSE={average.rmse:.3f}\n'
		'MAE={average.mae:.3f}\n'
		'Delta1={average.delta1:.3f}\n'
		'REL={average.absrel:.3f}\n'
		'Lg10={average.lg10:.3f}\n'
		't_GPU={time:.3f}\n'.format(
		average=avg, time=avg.gpu_time))

	if write_to_file:
		with open(test_csv, 'a') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
				'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
				'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
	return avg, img_merge

if __name__ == '__main__':
	main()
