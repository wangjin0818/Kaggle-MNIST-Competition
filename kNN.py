from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import util

def knnClassify(trainData, trainLabel, testData):
	knnClf = KNeighborsClassifier()
	knnClf.fit(trainData, np.ravel(trainLabel))
	testLabel = knnClf.predict(testData)
	util.saveResult(testLabel, './result/sklearn_knn_result.csv')
	return testLabel

if __name__ == "__main__":
	trainData, trainLabel = util.loadTrainData()
	testData = util.loadTestData()

	knnClassify(trainData, trainLabel, testData)
	print 'knnClassify is finished!'
