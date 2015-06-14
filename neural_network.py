import numpy as np
import util

import os
import gzip
import cPickle
import math
import urllib
import sys
import logging

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

def sigmoid_gradient(z):
	return sigmoid(z) * (1 - sigmoid(z))

def nnCostFunction(output, y, W_1, W_2, lamb=1):
	m, k = output.shape
	# print m, k 

	cost = 0.
	for i in range(m):
		for j in range(k):
			cost += -y[i, j] * math.log(output[i, j]) - (1 - y[i, j]) * math.log(1 - output[i, j])
	cost = cost / m

	theta_1 = 0.
	for i in range(W_1.shape[0]):
		for j in range(W_1.shape[1]):
			theta_1 += (W_1[i, j] ** 2)
	
	theta_2 = 0.
	for i in range(W_2.shape[0]):
		for j in range(W_2.shape[1]):
			theta_2 += (W_2[i, j] ** 2)

	return (cost + lamb * (theta_1 + theta_2) / (2 * m))

if __name__ == '__main__':
	program = os.path.basename(sys.argv[0])
	logger = logging.getLogger(program)
    
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)
	logger.info("running %s" % ''.join(sys.argv))

	# trainData, trainLabel = load_data('mnist.pkl.gz')
	trainData, trainLabel = util.load_total_data()
	testData = util.loadTestData()
	trainData = util.upToInt(trainData)

	print trainLabel
	y = np.zeros((trainLabel.shape[0], 10))
	for i in range(len(trainLabel)):
		y[i, trainLabel[i]] = 1

	rng = np.random.RandomState(0)
	# threshold of improvement, if improvement below this value, then
	# break the iteration
	impr_threshold = 0.001
	# total number of epoch
	n_epoch = 5000
	# lambda for regularization
	lamb = 0.01
	# learning rate
	alpha = 0.12
	# node number of hidden layer
	n_hiddenlayer = 500

	n_in = trainData.shape[1]
	n_out = 10

	W_1 = np.asarray(
		rng.uniform(
			low=-np.sqrt(6. / (n_in + n_hiddenlayer)),
			high=np.sqrt(6. / (n_in + n_hiddenlayer)),
			size=(n_in, n_hiddenlayer)
		)
	)

	b_1 = np.zeros((n_hiddenlayer, ))

	W_2 = np.asarray(
		rng.uniform(
			low=-np.sqrt(6. / (n_hiddenlayer + n_out)),
			high=np.sqrt(6. / (n_hiddenlayer + n_out)),
			size=(n_hiddenlayer, n_out)
		)
	)

	b_2 = np.zeros((n_out, ))

	prev_cost = np.inf
	for i in range(n_epoch):
		# feed forward propoagation
		a_1 = trainData
		z_2 = np.dot(a_1, W_1) + b_1
		a_2 = sigmoid(z_2)
		z_3 = np.dot(a_2, W_2) + b_2
		a_3 = sigmoid(z_3)

		cost = nnCostFunction(a_3, y, W_1, W_2)
		impr = prev_cost - cost
		logging.info('epoch %d, cost: %f, impr: %f' % (i+1, cost, impr))
		if impr < impr_threshold:
			break
		prev_cost = cost

		## back propagation
		## allocate for delta_W and delta_b
		delta_3 = a_3 - y
		delta_2 = np.multiply(np.dot(delta_3, W_2.transpose()), sigmoid_gradient(z_2))

		delta_W_2 = (np.dot(a_2.transpose(), delta_3)) / trainData.shape[0]
		delta_b_2 = np.sum(delta_3.transpose(), axis=1) / trainData.shape[0]

		W_2 = W_2 - alpha * (delta_W_2 + lamb * W_2)
		b_2 = b_2 - alpha * delta_b_2

		delta_W_1 = (np.dot(a_1.transpose(), delta_2)) / trainData.shape[0]
		delta_b_1 = np.sum(delta_2.transpose(), axis=1) / trainData.shape[0]

		W_1 = W_1 - alpha * (delta_W_1 + lamb * W_1)
		b_1 = b_1 - alpha * delta_b_1

	## prediction
	test_a_2 = sigmoid(np.dot(testData, W_1) + b_1)
	test_a_3 = sigmoid(np.dot(test_a_2, W_2) + b_2)

	logging.info('... saving data')
	testLabel = np.argmax(test_a_3, axis=1)
	util.saveResult(testLabel, './result/neural_network.csv')
	logging.info('the processing is finished!')