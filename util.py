import numpy as np
import csv
import gzip
import cPickle
import os

def loadTrainData(Norm=True):
	l = []
	with open('./data/train.csv') as file:
		lines = csv.reader(file)
		for line in lines:
			l.append(line)	# size 42001 * 785
	l.remove(l[0]) # remove header
	l = np.array(l)
	label = l[:, 0]
	data = l[:, 1:]
	if Norm:
		return nomalizing(dataToInt(data)), labelToInt(label)
	else:
		return dataToInt(data), labelToInt(label)

def dataToInt(array):
	array = np.mat(array)
	m, n = np.shape(array)
	newArray = np.zeros((m, n))
	for i in xrange(m):
		for j in xrange(n):
			newArray[i, j] = int(array[i, j])
	return newArray

def upToInt(array):
	array = np.mat(array)
	m, n = np.shape(array)
	newArray = np.zeros((m, n))
	for i in xrange(m):
		for j in xrange(n):
			if array[i, j] > 0.:
				newArray[i, j] = 1
	return newArray

def labelToInt(array):
	m = len(array)
	newArray = []
	for i in xrange(m):
		newArray.append(int(array[i]))
	return np.array(newArray)

def nomalizing(array):
	m, n = np.shape(array)
	for i in xrange(m):
		for j in xrange(n):
			if array[i,j] != 0:
				array[i, j] = 1
	return array

def loadTestData(Norm=True):
	l = []
	with open('./data/test.csv') as file:
		lines = csv.reader(file)
		for line in lines:
			l.append(line) # size 28001 * 784
	l.remove(l[0])
	data = np.array(l)
	if Norm:
		return nomalizing(dataToInt(data))
	else:
		return dataToInt(data)

def saveResult(result, csvName):
	with open(csvName, 'wb') as myFile:
		myWriter = csv.writer(myFile)
		myWriter.writerow(["ImageId", "Label"])
		for i in range(len(result)):
			tmp = []
			tmp.append(str(i+1))
			tmp.append(int(result[i]))
			myWriter.writerow(tmp)

def load_total_data(dataset='mnist.pkl.gz'):
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

	# Load the dataset
	f = gzip.open(dataset, 'rb')
	(trainData, trainLabel), (validData, validLabel), (testData, testLabel) \
								= cPickle.load(f)
	f.close()

	retData = np.vstack((trainData, validData, testData))
	retLabel = np.hstack((trainLabel, validLabel, testLabel))

	return retData, retLabel
