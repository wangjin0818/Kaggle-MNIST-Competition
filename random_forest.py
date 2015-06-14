from sklearn.ensemble import RandomForestClassifier

import numpy as np
import util

def rfClassify(trainData, trainLabel, testData):
	rfClf = RandomForestClassifier(n_estimators=100)
	rfClf.fit(trainData, np.ravel(trainLabel))
	testLabel = rfClf.predict(testData)
	util.saveResult(testLabel, './result/random_forest_result.csv')
	return testLabel

if __name__ == "__main__":
	trainData, trainLabel = util.loadTrainData()
	testData = util.loadTestData()

	rfClassify(trainData, trainLabel, testData)
	print 'rfClassify is finished!'
