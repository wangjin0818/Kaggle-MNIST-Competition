from sklearn import svm

import numpy as np
import pandas as pd

import util

def svcClassify(trainData, trainLabel, testData):
	svcClf = svm.SVC(C=5.0)
	svcClf.fit(trainData, np.ravel(trainLabel))
	testLabel = svcClf.predict(testData)
	util.saveResult(testLabel, './result/svm_result.csv')
	return testLabel

def svmLinearClassify(trainData, trainLabel,testData):
	svcClf = svm.SVC(kernel='linear')
	svcClf.fit(trainData, np.ravel(trainLabel))
	testLabel = svcClf.predict(testData)
	util.saveResult(testLabel, './result/svm_linear_result.csv')
	return testLabel

if __name__ == "__main__":
	trainData, trainLabel = util.loadTrainData()
	testData = util.loadTestData()

	svcClassify(trainData, trainLabel, testData)
	print 'svcClassify is finished!'

	svmLinearClassify(trainData, trainLabel, testData)
	print 'svm_linear is finished!'
