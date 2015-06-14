from sklearn import tree

import numpy as np
import util

def treeClassify(trainData, trainLabel, testData):
	treeClf = tree.DecisionTreeClassifier()
	treeClf.fit(trainData, np.ravel(trainLabel))
	testLabel = treeClf.predict(testData)
	util.saveResult(testLabel, './result/decision_tree_result.csv')
	return testLabel

if __name__ == "__main__":
	trainData, trainLabel = util.loadTrainData()
	testData = util.loadTestData()

	treeClassify(trainData, trainLabel, testData)
	print 'treeClassify is finished!'