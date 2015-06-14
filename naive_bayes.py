from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

import numpy as np
import util

def GaussianNBClassify(trainData, trainLabel, testData):
	nbClf = GaussianNB()
	nbClf.fit(trainData, np.ravel(trainLabel))
	testLabel = nbClf.predict(testData)
	util.saveResult(testLabel, './result/gaussion_nb_result.csv')
	return testLabel

def MultinomialNBClassify(trainData, trainLabel, testData):
	nbClf = MultinomialNB()
	nbClf.fit(trainData, np.ravel(trainLabel))
	testLabel = nbClf.predict(testData)
	util.saveResult(testLabel, './result/multinomial_nb_result.csv')
	return testLabel

if __name__ == "__main__":
	trainData, trainLabel = util.loadTrainData()
	testData = util.loadTestData()

	GaussianNBClassify(trainData, trainLabel, testData)
	print 'gaussion_nb is finished!'

	MultinomialNBClassify(trainData, trainLabel, testData)
	print 'multinomial_nb is finished!'

