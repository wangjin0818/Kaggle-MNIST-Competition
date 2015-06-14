from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import util

def baggingClassify(trainData, trainLabel, testData):
	baggingCif = BaggingClassifier(KNeighborsClassifier())
	baggingCif.fit(trainData, np.ravel(trainLabel))
	testLabel = baggingCif.predict(testData)
	util.saveResult(testLabel, './result/bagging_result.csv')
	return testLabel

if __name__ == "__main__":
	trainData, trainLabel = util.loadTrainData()
	testData = util.loadTestData()

	baggingClassify(trainData, trainLabel, testData)
	print 'baggingClassify is finished!'
