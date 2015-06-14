import gzip
import cPickle
import os

import numpy as np
import util

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
	(trainData, trainLabel), (validData, validLabel), (testData, testLabel) \
								= cPickle.load(f)
	f.close()

	retData = np.vstack((trainData, validData, testData))
	retLabel = np.hstack((trainLabel, validLabel, testLabel))

	return retData, retLabel

if __name__ == '__main__':
	trainData, trainLabel = load_data('mnist.pkl.gz')
	testData = util.loadTestData()
	trainData = util.upToInt(trainData)
	
	print trainData[1, :]
	print testData[1, :]
	
	
