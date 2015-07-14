import keras
import util
import os
import sys
import logging

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    # initialize constant variable
    batch_size = 128
    nb_classes = 10
    nb_epoch = 20

    logging.info("loading training data...")
    trainData, trainLabel = util.loadTrainData()
    trainData = trainData.astype("float32")
    trainLabel = np_utils.to_categorical(trainLabel, 10)

    logging.info("loading testing data...")
    testData = util.loadTestData()
    testData = testData.astype("float32")

    model = Sequential()
    model.add(Dense(784, 128, init="uniform"))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(128, 128, init="uniform"))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, 10, init="uniform"))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    model.fit(trainData, trainLabel, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)

    logging.info("Training process finished!")
    logging.info("Predict for testData...")
    y_pred = model.predict(testData,)
    testLabel = np.argmax(y_pred, axis=1)

    logging.info("Save result...")
    util.saveResult(testLabel, './result/keras_mlp_result.csv')