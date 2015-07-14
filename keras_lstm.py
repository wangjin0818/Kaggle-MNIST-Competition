import keras
import util
import os
import sys
import logging

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN, LSTM
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
    batch_size = 32
    nb_classes = 10
    nb_epoch = 200
    hidden_units = 100

    learning_rate = 1e-6
    clip_norm = 1.0
    BPTT_truncate = 28 * 28

    logging.info("loading training data...")
    trainData, trainLabel = util.loadTrainData()
    trainData = trainData.astype("float32")
    trainLabel = np_utils.to_categorical(trainLabel, 10)

    logging.info("loading testing data...")
    testData = util.loadTestData()
    testData = testData.astype("float32")

    trainData = trainData.reshape(trainData.shape[0], -1, 1)
    testData = testData.reshape(testData.shape[0], -1, 1)

    logging.info('Predicting with LSTM...')
    model = Sequential()
    model.add(LSTM(1, hidden_units))
    model.add(Dense(hidden_units, nb_classes))
    model.add(Activation('softmax'))
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

    model.fit(trainData, trainLabel, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)

    logging.info("Training process finished!")
    logging.info("Predict for testData...")
    y_pred = model.predict(testData,)
    testLabel = np.argmax(y_pred, axis=1)

    logging.info("Save result...")
    util.saveResult(testLabel, './result/keras_lstm_result.csv')