
from sklearn.ensemble import AdaBoostClassifier

import numpy as np
import util

def adaboostClassify(trainData, trainLabel, testData):
	abClf = AdaBoostClassifier(n_estimators=500)
	abClf.fit(trainData, np.ravel(trainLabel))
	testLabel = abClf.predict(testData)
	util.saveResult(testLabel, './result/adaboost_result.csv')
	return testLabel

if __name__ == "__main__":
	trainData, trainLabel = util.loadTrainData()
	testData = util.loadTestData()

	adaboostClassify(trainData, trainLabel, testData)
	print 'adaboostClassify is finished!'
