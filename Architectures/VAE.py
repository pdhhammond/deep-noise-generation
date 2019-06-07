import torch
from torch import nn
from torch.autograd.variable import Variable

def normal_init(m, mean, std):
	if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
		m.weight.data.normal_(mean, std)
		m.bias.data.zero_()

class VariationalEncoderNet(torch.nn.Module):
	def __init__(self, encoded_dims=100):
		super(VariationalEncoderNet, self).__init__()
		filters = 32
		c_in = 2

		# downsample
		self.hidden0 = nn.Sequential(
			nn.Conv2d(c_in, filters, 8, 2, 3),
			nn.LeakyReLU(.2)
		)
		self.hidden1 = nn.Sequential(
			nn.Conv2d(filters, filters * 2, 8, 2, 3),
			nn.BatchNorm2d(filters * 2),
			nn.LeakyReLU(.2)
		)
		self.hidden2 = nn.Sequential(
			nn.Conv2d(filters * 2, filters * 4, 8, 2, 3),
			nn.BatchNorm2d(filters * 4),
			nn.LeakyReLU(.2)
		)
		self.hidden3 = nn.Sequential(
			nn.Conv2d(filters * 4, filters * 8, 8, 2, 3),
			nn.BatchNorm2d(filters * 8),
			nn.LeakyReLU(.2)
		)
		self.hidden4 = nn.Sequential(
			nn.Conv2d(filters * 8, filters * 16, 8, 2, 3),
			nn.BatchNorm2d(filters * 16),
			nn.LeakyReLU(.2)
		)
		self.mu_out = nn.Sequential(
			nn.Conv2d(filters * 16, encoded_dims, 8, 1, 0),
			nn.Tanh()
		)
		self.sigma_out = nn.Sequential(
			nn.Conv2d(filters * 16, encoded_dims, 8, 1, 0),
			nn.Tanh()
		)

	def weight_init(self, mean, std):
		for m in self._modules:
			normal_init(self._modules[m], mean, std)

	def forward(self, input):
		# downsample
		x = self.hidden0(input)
		x = self.hidden1(x)
		x = self.hidden2(x)
		x = self.hidden3(x)
		x = self.hidden4(x)

		# output estimated mean and standard deviation
		mu = self.mu_out(x)
		sigma = self.sigma_out(x)
		return mu, sigma

class VariationalDecoderNet(torch.nn.Module):
	def __init__(self, encoded_dims=100):
		super(VariationalDecoderNet, self).__init__()
		filters = 32
		# for the RGB-D input, we actually take a grayscale image with a depth
		# image, so it's only a 2-channel input
		c_in = 2

		# conditional input stream
		# downsamples the input RGB-D image to the dimensions of the
		# up-projected z-vector before concatenation
		self.label_input = nn.Sequential(
			nn.Conv2d(c_in, filters, 8, 2, 3),
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

		# embedded vector input stream
		# up-projects the input z vector
		self.unembedder = nn.Sequential(
			nn.ConvTranspose2d(encoded_dims, filters * 8, 8, 1, 0),
			nn.BatchNorm2d(filters * 8),
			nn.LeakyReLU(.2),
		)

		# decoder sctructure
		# up-samples the concatenated up-projected z-vector and down-sampled
		# conditional inputs
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

		# dropout output stream
		self.hidden4_dropout = nn.Sequential(
			nn.ConvTranspose2d(filters * 2, filters, 8, 2, 3),
			nn.BatchNorm2d(filters),
			nn.LeakyReLU(.2),
		)
		self.out_dropout = nn.Sequential(
			nn.ConvTranspose2d(filters, 1, 8, 2, 3),
			nn.Tanh()
		)

		# noise output stream
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

	# performs the reparametrization trick for the forward pass
	# using the mean and standard-dev vectors from the encoder
	def reparametrize(self, mu, sigma):
		std = sigma.mul(.5).exp_()

		# check if the GPU is in use
		if torch.cuda.is_available():
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()

		eps = Variable(eps)
		return eps.mul(std).add_(mu)

	# peforms the forward pass ignoring the reparametrization step
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

	# performs the forward pass using the reparametrization step
	def forward(self, mu, sigma, label):
		# reparametrization trick
		z = self.reparametrize(mu, sigma)
		# ensures the embedded vector is unit-length
		z = nn.functional.normalize(z, p=2, dim=1)

		x = self.generate(z, label)
		return x