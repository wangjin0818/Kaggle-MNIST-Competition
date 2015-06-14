from sklearn import linear_model

import numpy as np
import util

def lrClassify(trainData, trainLabel, testData):
	lrClf = linear_model.LogisticRegression(C=1e5)
	lrClf.fit(trainData, np.ravel(trainLabel))
	testLabel = lrClf.predict(testData)
	util.saveResult(testLabel, './result/logistic_regression_result.csv')
	return testLabel

if __name__ == "__main__":
	trainData, trainLabel = util.loadTrainData()
	testData = util.loadTestData()

	lrClassify(trainData, trainLabel, testData)
	print 'logistic_regression is finished!'