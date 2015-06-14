import numpy as np
import util

from nolearn.dbn import DBN

if __name__ == "__main__":
	trainData, trainLabel = util.load_total_data()
	testData = util.loadTestData()
	trainData = util.upToInt(trainData)

	dbn = DBN(
		[trainData.shape[1], 300, 10],
		learn_rates = 0.3,
		learn_rate_decays = 0.9,
		epochs = 100,
		verbose = 1)
	
	dbn.fit(trainData, np.ravel(trainLabel))

	testLabel = dbn.predict(testData)
	util.saveResult(testLabel, './result/dbn_result.csv')