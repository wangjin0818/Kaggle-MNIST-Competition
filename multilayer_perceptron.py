"""
the multilayer perceptron using Theano
"""

import os
import sys
import time
import gzip
import cPickle

import numpy as np

import theano
import theano.tensor as T

import util

class LogisticRegression(object):
	"""
	Multi-class Logitic Regression Class
	"""
	def __init__(self, input, n_in, n_out):
		# assign spacing for W and b
		self.W = theano.shared(
			value = np.zeros(
				(n_in, n_out),
				dtype = theano.config.floatX
			),
			name = 'W',
			borrow = True
		)

		# initialize the baises b as a vector of n_out 0s
		self.b = theano.shared(
			value = np.zeros(
				(n_out, ),
				dtype = theano.config.floatX
			),
			name = 'b',
			borrow = True
		)

		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

		# parameters of the model
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y):
		"""
		Return the mean of the negative log-likelihood of the prediction
		of this model under a given target distribution.
		"""
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		"""

		"""

		# check if y has same dimension of y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		# check if y is of the correct datatype
		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1
			# represents a mistake in prediction
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()

class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,
		activation=T.tanh):
		self.input = input

		# if W and b is none, assign random value for them
		if W is None:
			W_values = np.asarray(
				# assign uniform distribution for W_value
				rng.uniform(
						low=-np.sqrt(6. / (n_in + n_out)),
						high=np.sqrt(6. / (n_in + n_out)),
						size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
			# if activation is sigmoid function, then low and high boundary
			# of W_value will be multiply 4 times
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value=W_values, name='W', borrow=True)

		# as same as b
		if b is None:
			b_values = np.zeros((n_out, ), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		# combine the output of the model
		# y = activation(W * X + b)
		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)

		# parameters of the model
		self.params = [self.W, self.b]

class MLP(object):
	"""
	Multi-Layer Perceptron Class
	"""
	def __init__(self, rng, input, n_in, n_hidden, n_out):
		# hidden layer
		self.hiddenLayer = HiddenLayer(
			rng = rng,
			input = input,
			n_in = n_in,
			n_out = n_hidden,
			activation = T.tanh
		)

		# The logistic regression layer gets as input the hidden units
		# of the hidden layer
		self.logRegressionLayer = LogisticRegression(
			input = self.hiddenLayer.output,
			n_in = n_hidden,
			n_out = n_out
		)

		# L1 regularization
		self.L1 = (
			abs(self.hiddenLayer.W).sum()
			+ abs(self.logRegressionLayer.W).sum()
		)

		# L2 regularization
		self.L2_sqr = (
			(self.hiddenLayer.W ** 2).sum()
			+ (self.logRegressionLayer.W ** 2).sum()
		)

		self.negative_log_likelihood = (
			self.logRegressionLayer.negative_log_likelihood
		)

		self.errors = self.logRegressionLayer.errors

		self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def shared_data(data_xy, borrow=True):
	data_x, data_y = data_xy
	
	shared_x = theano.shared(np.asarray(data_x, 
		dtype = theano.config.floatX),
		borrow = borrow
	)

	shared_y = theano.shared(np.asarray(data_y,
		dtype = theano.config.floatX),
		borrow = borrow
	)

	return shared_x, T.cast(shared_y, 'int32')

def load_data(dataset):
	# load data
	data_dir, data_file = os.path.split(dataset)
	if data_dir == "" and not os.path.isfile(dataset):
		# check if dataset is in the data directory,
		new_path = os.path.join(
			".",
			"data",
			dataset
		)
		if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
			dataset = new_path

	if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
		import urllib
		origin = (
			'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
		)
		print 'Downloading data from %s' % origin
		urllib.urlretrieve(origin, dataset)

	print '... loading data'

	# Load the dataset
	f = gzip.open(dataset, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	print '[*] train_set sample'
	print train_set
	print '[*] test_set sample'
	print test_set

	test_set_x, test_set_y = shared_data(test_set)
	valid_set_x, valid_set_y = shared_data(valid_set)
	train_set_x, train_set_y = shared_data(train_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
		(test_set_x, test_set_y)]
	return rval


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
	dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
	
	# load data from train dataset
	'''
	trainData, trainLabel = util.loadTrainData()
	testData = util.loadTestData()

	n_train_batches = trainData.shape[0] / batch_size
	n_test_batches = testData.shape[0] / batch_size

	print n_train_batches, n_test_batches
	'''
	datasets = load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'

	# allocate symbolic variables for the data
	index = T.lscalar() # index to a [mini]batch
	x = T.matrix('x') # the data is presented as tasterized images
	y = T.ivector('y') 	# the labels are presented as 1D vector of 
						# [int] labels

	rng = np.random.RandomState(1234)

	# construct the MLP class
	classifier = MLP(
		rng = rng,
		input = x,
		n_in = 28 * 28,
		n_hidden = n_hidden,
		n_out = 10
	)

	cost = (
		classifier.negative_log_likelihood(y)
		+ L1_reg * classifier.L1
		+ L2_reg * classifier.L2_sqr
	)
	
	test_model = theano.function(
		inputs = [index],
		outputs = classifier.errors(y),
		givens = {
			x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)
	
	validate_model = theano.function(
		inputs = [index],
		outputs = classifier.errors(y),
		givens = {
			x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	gparams = [T.grad(cost, param) for param in classifier.params]

	updates = [
		(param, param - learning_rate * gparams)
		for param, gparams in zip(classifier.params, gparams)
	]

	## train model definition
	train_model = theano.function(
		inputs = [index],
		outputs = cost,
		updates = updates,
		givens = {
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	###############
	# TRAIN MODEL #
	###############
	print '... training'

	patience = 10000 # look as this many examples regardless
	patience_increase = 2 	# wait this much longer when a new best is
							# found
	improvement_threshold = 0.995 	# a relative improvement of this much is
									# considered significant
	validation_frequency = min(n_train_batches, patience / 2)

	best_validation_loss = np.inf
	best_iter = 0
	test_score = 0.
	start_time = time.clock()

	epoch = 0
	done_looping = False

	while(epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):

			minibatch_avg_cost = train_model(minibatch_index)
			# iteration number
			iter = (epoch - 1) * n_train_batches + minibatch_index

			if (iter + 1) % validation_frequency == 0:
				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i
								in xrange(n_valid_batches)]
				this_validation_loss = np.mean(validation_losses)

				print(
					'epoch %i, minibatch %i/%i, validation error %f %%' %
					(
						epoch,
						minibatch_index + 1,
						n_train_batches,
						this_validation_loss * 100.
					)
				)

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					# improve patience if loss improvement is good enough
					if (
						this_validation_loss < best_validation_loss *
						improvement_threshold
					):
						patience = max(patience, iter * patience_increase)

					best_validation_loss = this_validation_loss
					best_iter = iter

					# test it on the test set
					test_losses = [test_model(i) for i in xrange(n_test_batches)]
					test_score = np.mean(test_losses)

					print(('     epoch %i, minibatch %i/%i, test error of '
							'best model %f %%') %
							(epoch, minibatch_index + 1, n_train_batches,
							test_score * 100.))

				if patience <= iter:
					done_looping = True
					break

		end_time = time.clock()
		print(('Optimization complete. Best validation score of %f %% '
			'obtained at iteration %i, with test performance %f %%') %
			(best_validation_loss * 100., best_iter + 1, test_score * 100.))			

		print >> sys.stderr, ('The code for file ' +
						os.path.split(__file__)[1] +
						' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == "__main__":
	test_mlp()