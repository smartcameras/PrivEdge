from __future__ import print_function, division
import scipy
import errno
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from load_data import DataLoader
import numpy as np
import os


class RAN():
	def __init__(self, identity):
		self.identity = identity
		self.img_rows = 32
		self.img_cols = 32
		self.channels = 3
		self.img_shape = (self.img_rows, self.img_cols, self.channels)

		# Configure data loader
		self.dataset_name = self.identity
		self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))
		

		# Calculate output shape of D (PatchRAN)
		patch = int(self.img_rows / 2**4)
		self.disc_patch = (patch, patch, 1)
		
		optimizer = Adam(0.0002, 0.5)


		# Number of filters in the first layer of G and D
		self.gf = 32
		self.df = 64
		
		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='mse',
			optimizer=optimizer,
			metrics=['accuracy'])

		# Build and compile the reconstructor
		self.reconstructor = self.build_reconstructor()
		print(self.reconstructor.summary())
		self.reconstructor.compile(loss='mse', optimizer=optimizer)

		# The reconstructor takes noise as input and generated imgs
		img = Input(shape=self.img_shape)
		reconstr = self.reconstructor(img)

		# For the combined model we will only train the reconstructor
		self.discriminator.trainable = False


		# The valid takes generated images as input and determines validity
		valid = self.discriminator(reconstr)

		# The combined model  (stacked reconstructor and discriminator) takes
		# images as input => reconstruct images => determines validity
		self.combined = Model(img, [reconstr,valid])
		self.combined.compile(loss=['mse','mse'], loss_weights=[0.999, 0.001], optimizer=optimizer)

	def build_reconstructor(self):
			"""reconstructor"""

			def conv2d(layer_input, filters, f_size=4):
				"""Layers used during downsampling"""
				d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
				d = LeakyReLU(alpha=0.2)(d)
				d = InstanceNormalization()(d)
				return d

			def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
				"""Layers used during upsampling"""
				u = UpSampling2D(size=2)(layer_input)
				u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
				if dropout_rate:
					u = Dropout(dropout_rate)(u)
				u = InstanceNormalization()(u)
				return u

			# Image input
			d0 = Input(shape=self.img_shape)

			# Downsampling
			d1 = conv2d(d0, self.gf)
			d2 = conv2d(d1, self.gf*2)
			d3 = conv2d(d2, self.gf*4)
			d4 = conv2d(d3, self.gf*4)
			d5 = conv2d(d4, self.gf*8)

			# Upsampling
			u1 = deconv2d(d5, self.gf*8)
			u2 = deconv2d(u1, self.gf*8)
			u3 = deconv2d(u2, self.gf*8)
			u4 = deconv2d(u3, self.gf*4)
			u5 = deconv2d(u4, self.gf*2)

			output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u5)
			return Model(d0, output_img)

	def build_discriminator(self):

		def d_layer(layer_input, filters, f_size=4, normalization=True):
			"""Discriminator layer"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if normalization:
				d = InstanceNormalization()(d)
			return d

		img = Input(shape=self.img_shape)

		d1 = d_layer(img, self.df, normalization=False)
		d2 = d_layer(d1, self.df*2)
		d3 = d_layer(d2, self.df*4)
		d4 = d_layer(d3, self.df*8)

		validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

		return Model(img, validity)

	def train(self, epochs, batch_size=128, save_interval=50):
		
		half_batch = int(batch_size / 2)

		start_time = datetime.datetime.now()
		
		
		imgsVal= self.data_loader.load_data(self.identity,batch_size=half_batch,is_testing=True)
		TrainLoss = np.zeros(epochs)
		ValLoss  = np.ones(epochs)
		for epoch in range(epochs):

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# Sample reconstructor input
			img= self.data_loader.load_data(self.identity,batch_size=half_batch)
			
			# Reconstruct a batch of new images
			reconstr = self.reconstructor.predict(img)
			
			# Adversarial loss ground truths
			valid = np.ones((half_batch,) + self.disc_patch)
			fake = np.zeros((half_batch,) + self.disc_patch)
			

			# Train the discriminator
			d_loss_real = self.discriminator.train_on_batch(img, valid)
			d_loss_fake = self.discriminator.train_on_batch(reconstr, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			
			# ---------------------
			#  Train reconstructor
			# ---------------------

			# Sample reconstructor input
			img= self.data_loader.load_data(self.identity,batch_size=half_batch)
			
			# Train the reconstructor
			r_loss = self.combined.train_on_batch(img, [img,valid])
			r_loss_val= self.combined.test_on_batch(imgsVal, [imgsVal,valid])
			TrainLoss[epoch] = r_loss[0]
			MinValLoss = ValLoss.min()
			ValLoss[epoch] = r_loss_val[0]
			
			# Plot the progress
			print ("%d [D loss: %f, acc.: %.2f%%] [R loss: %f] [R loss Val: %f] [Minimum: %f]" % (epoch, d_loss[0], 100*d_loss[1], r_loss[0], r_loss_val[0], MinValLoss))

			# If at save interval => save generated image samples
			if ValLoss[epoch] < MinValLoss and MinValLoss < 0.04 :
				self.save_imgs(epoch)
				self.reconstructor.save('SavedModel/%s/%s.h5'%(self.identity,self.identity))
		np.savez('loss/Loss_%s'%(self.dataset_name), TrLoss=TrainLoss, TeLoss=ValLoss)        
	
	def save_imgs(self, epoch):
		r, c = 2, 2

		imgs= self.data_loader.load_data( self.identity,batch_size=1, is_testing=False)
		imgs_val = self.data_loader.load_data( self.identity,batch_size=1, is_testing=True)


		# Translate images to the other domain
		reconstr = self.reconstructor.predict(imgs)
		reconstr_val = self.reconstructor.predict(imgs_val)
		
		gen_imgs = np.concatenate([imgs, imgs_val, reconstr, reconstr_val])

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		titles = ['Train', 'Val', 'Reconstructed']
		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt])
				axs[i, j].set_title(titles[j])
				axs[i,j].axis('off')
				cnt += 1       
		fig.savefig("ReconstructedImages/%s/TrainValSamples_E%d.png" % (self.dataset_name, epoch))
		plt.close()


