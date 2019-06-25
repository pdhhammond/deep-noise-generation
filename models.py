"""
From https://github.com/fangchangma/sparse-to-dense.pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math

class VariationalDecoderNet(torch.nn.Module):
	def __init__(self, encoded_dims=100):
		super(VariationalDecoderNet, self).__init__()
		filters = 32
		c_out = 2

		# conditional input
		self.label_input = nn.Sequential(
			nn.Conv2d(2, filters, 8, 2, 3),
			nn.LeakyReLU(.2),
			nn.Conv2d(filters, filters * 2, 8, 2, 3),
			nn.BatchNorm2d(filters * 2),
			nn.LeakyReLU(.2),
			nn.Conv2d(filters * 2, filters * 4, 8, 2, 3),
			nn.BatchNorm2d(filters * 4),
			nn.LeakyReLU(.2),
			nn.Conv2d(filters * 4, filters * 8, 8, 2, 3),
			nn.BatchNorm2d(filters * 8),
			nn.LeakyReLU(.2),
			nn.Conv2d(filters * 8, filters * 8, 8, 2, 3),
			nn.BatchNorm2d(filters * 8),
			nn.LeakyReLU(.2)
		)

		# upsample
		self.unembedder = nn.Sequential(
			nn.ConvTranspose2d(encoded_dims, filters * 8, 8, 1, 0),
			nn.BatchNorm2d(filters * 8),
			nn.LeakyReLU(.2),
		)
		self.hidden1 = nn.Sequential(
			nn.ConvTranspose2d(filters * 16, filters * 8, 8, 2, 3),
			nn.BatchNorm2d(filters * 8),
			nn.LeakyReLU(.2),
		)
		self.hidden2 = nn.Sequential(
			nn.ConvTranspose2d(filters * 8, filters * 4, 8, 2, 3),
			nn.BatchNorm2d(filters * 4),
			nn.LeakyReLU(.2),
		)
		self.hidden3 = nn.Sequential(
			nn.ConvTranspose2d(filters * 4, filters * 2, 8, 2, 3),
			nn.BatchNorm2d(filters * 2),
			nn.LeakyReLU(.2),
		)
		self.hidden4_dropout = nn.Sequential(
			nn.ConvTranspose2d(filters * 2, filters, 8, 2, 3),
			nn.BatchNorm2d(filters),
			nn.LeakyReLU(.2),
		)
		self.out_dropout = nn.Sequential(
			nn.ConvTranspose2d(filters, 1, 8, 2, 3),
			nn.Tanh()
		)
		self.hidden4_noise = nn.Sequential(
			nn.ConvTranspose2d(filters * 2, filters, 8, 2, 3),
			nn.BatchNorm2d(filters),
			nn.LeakyReLU(.2),
		)
		self.out_noise = nn.Sequential(
			nn.ConvTranspose2d(filters, 1, 8, 2, 3),
			nn.Tanh()
		)

	def weight_init(self, mean, std):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)

	def reparametrize(self, mu, sigma):
		std = sigma.mul(.5).exp_()
		if torch.cuda.is_available():
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mu)

	def generate(self, z, label):
		x0 = self.unembedder(z)
		x1 = self.label_input(label)
		x = torch.cat([x0, x1], 1)

		x = self.hidden1(x)
		x = self.hidden2(x)
		x = self.hidden3(x)
		x0 = self.hidden4_dropout(x)
		x0 = self.out_dropout(x0)
		x1 = self.hidden4_noise(x)
		x1 = self.out_noise(x1)
		x = torch.cat([x0, x1], 1)
		return x

	def forward(self, mu, sigma, label):
		# reparametrization trick
		z = self.reparametrize(mu, sigma)
		z = nn.functional.normalize(z, p=2, dim=1)

		x = self.generate(z, label)
		return x

class Unpool(nn.Module):
	# Unpool: 2*2 unpooling with zero padding
	def __init__(self, num_channels, stride=2):
		super(Unpool, self).__init__()

		self.num_channels = num_channels
		self.stride = stride

		# create kernel [1, 0; 0, 0]
		self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda()) # currently not compatible with running on CPU
		self.weights[:,:,0,0] = 1

	def forward(self, x):
		return F.conv_transpose2d(x, self.weights, stride=self.stride, groups=self.num_channels)

def weights_init(m):
	# Initialize filters with Gaussian random weights
	if isinstance(m, nn.Conv2d):
		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
		if m.bias is not None:
			m.bias.data.zero_()
	elif isinstance(m, nn.ConvTranspose2d):
		n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
		if m.bias is not None:
			m.bias.data.zero_()
	elif isinstance(m, nn.BatchNorm2d):
		m.weight.data.fill_(1)
		m.bias.data.zero_()

class Decoder(nn.Module):
	# Decoder is the base class for all decoders

	names = ['deconv2', 'deconv3', 'deconv5', 'deconv8', 'upconv', 'upproj']

	def __init__(self):
		super(Decoder, self).__init__()

		self.layer1 = None
		self.layer2 = None
		self.layer3 = None
		self.layer4 = None
		self.layer5 = None

		self.skip4_pad = nn.ZeroPad2d((1,0,1,0))
		self.skip3_pad = nn.ZeroPad2d((1,1,2,1))
		self.skip2_pad = nn.ZeroPad2d((2,2,4,3))
		self.skip1_pad = nn.ZeroPad2d((4,4,7,7))

	def forward(self, x, skip1=None, skip2=None, skip3=None, skip4=None):
		skip4 = self.skip4_pad(skip4)
		skip3 = self.skip3_pad(skip3)
		skip2 = self.skip2_pad(skip2)
		skip1 = self.skip1_pad(skip1)


		x = self.layer1(x)	# connects to skip4
		x = torch.cat((x, skip4), 1)

		x = self.layer2(x)	# connects to skip3
		x = torch.cat((x, skip3), 1)

		x = self.layer3(x)	# connects to x skip2
		x = torch.cat((x, skip2), 1)

		x = self.layer4(x)
		x = torch.cat((x, skip1), 1)
		x = self.layer5(x)
		return x

class DeConv(Decoder):
	def __init__(self, in_channels, kernel_size):
		assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
		super(DeConv, self).__init__()

		def convt(in_channels, out_channels=None):
			stride = 2
			padding = (kernel_size - 1) // 2
			output_padding = kernel_size % 2
			assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

			module_name = "deconv{}".format(kernel_size)
			out_channels = in_channels//2 if out_channels is None else out_channels
			return nn.Sequential(collections.OrderedDict([
				  (module_name, nn.ConvTranspose2d(in_channels,out_channels,kernel_size,
						stride,padding,output_padding,bias=False)),
				  ('batchnorm', nn.BatchNorm2d(out_channels)),
				  ('relu',      nn.ReLU(inplace=True)),
				]))

		self.layer1 = convt(in_channels, in_channels // 2)
		self.layer2 = convt(in_channels, in_channels // (2 ** 2))
		self.layer3 = convt(in_channels // 2, in_channels // (2 ** 3))
		self.layer4 = convt(in_channels // (2 ** 2), in_channels // (2 ** 3))
		self.layer5 = convt(in_channels // (2 ** 2), in_channels // (2 ** 4))
		# self.layer3 = convt(in_channels // (2 ** 2))
		# self.layer4 = convt(in_channels // (2 ** 3))
		# self.layer5 = convt(in_channels // (2 ** 4))

class UpConv(Decoder):
	# UpConv decoder consists of 4 upconv modules with decreasing number of channels and increasing feature map size
	def upconv_module(self, in_channels):
		# UpConv module: unpool -> 5*5 conv -> batchnorm -> ReLU
		upconv = nn.Sequential(collections.OrderedDict([
		  ('unpool',    Unpool(in_channels)),
		  ('conv',      nn.Conv2d(in_channels,in_channels//2,kernel_size=5,stride=1,padding=2,bias=False)),
		  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
		  ('relu',      nn.ReLU()),
		]))
		return upconv

	def __init__(self, in_channels):
		super(UpConv, self).__init__()
		self.layer1 = self.upconv_module(in_channels)
		self.layer2 = self.upconv_module(in_channels//2)
		self.layer3 = self.upconv_module(in_channels//4)
		self.layer4 = self.upconv_module(in_channels//8)

class UpProj(Decoder):
	# UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

	class UpProjModule(nn.Module):
		# UpProj module has two branches, with a Unpool at the start and a ReLu at the end
		#   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
		#   bottom branch: 5*5 conv -> batchnorm

		def __init__(self, in_channels):
			super(UpProj.UpProjModule, self).__init__()
			out_channels = in_channels//2
			self.unpool = Unpool(in_channels)
			self.upper_branch = nn.Sequential(collections.OrderedDict([
			  ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
			  ('batchnorm1', nn.BatchNorm2d(out_channels)),
			  ('relu',      nn.ReLU()),
			  ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
			  ('batchnorm2', nn.BatchNorm2d(out_channels)),
			]))
			self.bottom_branch = nn.Sequential(collections.OrderedDict([
			  ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
			  ('batchnorm', nn.BatchNorm2d(out_channels)),
			]))
			self.relu = nn.ReLU()

		def forward(self, x):
			x = self.unpool(x)
			x1 = self.upper_branch(x)
			x2 = self.bottom_branch(x)
			x = x1 + x2
			x = self.relu(x)
			return x

	def __init__(self, in_channels):
		super(UpProj, self).__init__()
		self.layer1 = self.UpProjModule(in_channels)
		self.layer2 = self.UpProjModule(in_channels//2)
		self.layer3 = self.UpProjModule(in_channels//4)
		self.layer4 = self.UpProjModule(in_channels//8)

def choose_decoder(decoder, in_channels):
	# iheight, iwidth = 10, 8
	if decoder[:6] == 'deconv':
		assert len(decoder)==7
		kernel_size = int(decoder[6])
		return DeConv(in_channels, kernel_size)
	elif decoder == "upproj":
		return UpProj(in_channels)
	elif decoder == "upconv":
		return UpConv(in_channels)
	else:
		assert False, "invalid option for decoder: {}".format(decoder)


class ResNet(nn.Module):
	def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

		if layers not in [18, 34, 50, 101, 152]:
			raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

		super(ResNet, self).__init__()
		pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

		if in_channels == 3:
			self.conv1 = pretrained_model._modules['conv1']
			self.bn1 = pretrained_model._modules['bn1']
		else:
			self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
			self.bn1 = nn.BatchNorm2d(64)
			weights_init(self.conv1)
			weights_init(self.bn1)

		self.output_size = output_size

		self.relu = pretrained_model._modules['relu']
		self.maxpool = pretrained_model._modules['maxpool']
		self.layer1 = pretrained_model._modules['layer1']
		self.layer2 = pretrained_model._modules['layer2']
		self.layer3 = pretrained_model._modules['layer3']
		self.layer4 = pretrained_model._modules['layer4']

		# clear memory
		del pretrained_model

		# define number of intermediate channels
		if layers <= 34:
			num_channels = 512
		elif layers >= 50:
			num_channels = 2048

		# self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
		# self.bn2 = nn.BatchNorm2d(num_channels//2)
		# self.decoder = choose_decoder(decoder, num_channels//2)
		self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size=1,bias=False)
		self.bn2 = nn.BatchNorm2d(num_channels)
		self.decoder = choose_decoder(decoder, num_channels)

		# setting bias=true doesn't improve accuracy
		# self.conv3 = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv3 = nn.Conv2d(num_channels//16,1,kernel_size=3,stride=1,padding=1,bias=False)
		# self.conv3 = nn.Conv2d(num_channels//64,1,kernel_size=3,stride=1,padding=1,bias=False)
		self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

		# weight init
		self.conv2.apply(weights_init)
		self.bn2.apply(weights_init)
		self.decoder.apply(weights_init)
		self.conv3.apply(weights_init)

	def forward(self, x):
		# resnet
		x1 = self.conv1(x)
		x = self.bn1(x1)
		x = self.relu(x)
		x = self.maxpool(x)
		x2 = self.layer1(x)
		x3 = self.layer2(x2)
		x4 = self.layer3(x3)
		x = self.layer4(x4)

		x = self.conv2(x)
		x = self.bn2(x)

		# decoder
		x = self.decoder(x, x1, x2, x3, x4)
		x = self.conv3(x)
		x = self.bilinear(x)
		return x
