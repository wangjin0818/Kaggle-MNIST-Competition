import keras
import util
import os
import sys
import logging

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, MaxPooling2D

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

    trainData = trainData.reshape(trainData.shape[0], 1, 28, 28)
    testData = testData.reshape(testData.shape[0], 1, 28, 28)

    model = Sequential()

    model.add(Convolution2D(32, 1, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(32*196, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    model.fit(trainData, trainLabel, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)

    logging.info("Training process finished!")
    logging.info("Predict for testData...")
    y_pred = model.predict(testData,)
    testLabel = np.argmax(y_pred, axis=1)

    logging.info("Save result...")
    util.saveResult(testLabel, './result/keras_cnn_result.csv')