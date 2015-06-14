import os
import sys
import logging
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

	def predict(self):
		"""
		Return the prediction of this model under a given target distribution.
		"""
		return self.y_pred

	def errors(self, y):
		"""
		Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
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

		self.predict = self.logRegressionLayer.predict

		self.params = self.hiddenLayer.params + self.logRegressionLayer.params

if __name__ == "__main__":
	program = os.path.basename(sys.argv[0])
	logger = logging.getLogger(program)
    
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)
	logger.info("running %s" % ''.join(sys.argv))

	logging.info('... loading data')
	trainData, trainLabel = util.load_total_data()
	testData = util.loadTestData()
	trainData = util.upToInt(trainData)

	## some parameters for training
	learning_rate = 0.12
	L1_reg = 0.001
	L2_reg = 0.0001
	n_hidden=500

	train_set_x = theano.shared(np.asarray(trainData,
		dtype = theano.config.floatX),
		borrow = True)

	train_set_y = theano.shared(np.asarray(trainLabel,
		dtype = theano.config.floatX),
		borrow = True)

	test_set_x = theano.shared(np.asarray(testData,
		dtype = theano.config.floatX),
		borrow = True)

	train_set_y = T.cast(train_set_y, 'int32')
	logging.info('... building the model')

	x = T.matrix('x') # the data is presented as tasterized images
	y = T.ivector('y') 	# the labels are presented as 1D vector of 
						# [int] labels

	rng = np.random.RandomState(1234)

	classifier = MLP(
		rng=rng, 
		input=x, 
		n_in=28*28, 
		n_hidden=n_hidden, 
		n_out=10
	)

	cost = (
		classifier.negative_log_likelihood(y)
		+ L1_reg * classifier.L1
		+ L2_reg * classifier.L2_sqr
	)

	gparams = [T.grad(cost, param) for param in classifier.params]

	updates = [
		(param, param - learning_rate * gparams)
		for param, gparams in zip(classifier.params, gparams)
	]

	## train model definition
	train_model = theano.function(
		inputs = [],
		outputs = cost,
		updates = updates,
		givens = {
			x: train_set_x,
			y: train_set_y
		}
	)

	logging.info('... training')
	improvement_threshold = 0.001
	epoch = 0
	n_epochs = 1000
	done_looping = False
	prev_cost = np.inf

	start_time = time.clock()
	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1

		cost = train_model()
		impr = prev_cost - cost
		logging.info('epoch %d, cost: %f, impr: %f' % (epoch+1, cost, impr))
		if impr < improvement_threshold:
			break
		prev_cost = cost

	end_time = time.clock()
	print >> sys.stderr, ('The code for file ' +
			os.path.split(__file__)[1] +
				' ran for %.2fm' % ((end_time - start_time) / 60.))

	# make a prediction
	predict_model = theano.function(
		inputs=[],
		outputs= classifier.predict(),
		givens={
			x: test_set_x
		}
	)

	testLabel = predict_model()
	util.saveResult(testLabel, './result/mlp_result.csv')