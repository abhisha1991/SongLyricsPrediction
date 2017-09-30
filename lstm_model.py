import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from SongLyricsPrediction.constants import *
import os

c = Constants()


def lstm_lyrics_generator(raw_text, dataX, dataY):

    chars = sorted(list(set(raw_text)))
    n_chars = len(raw_text)
    n_vocab = len(chars)
    n_patterns = len(dataX)
    int_to_char = dict((i, ch) for i, ch in enumerate(chars))

    # summarize data
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)
    print("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, c.lstm_seq_length, 1))

    # normalize
    X = X / float(n_vocab)

    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # define the lstm model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # define the checkpoint
    filepath = "songlyrics/lstm_saved.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # fit the model
    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

    # load the network weights
    model.load_weights(filepath)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

    # store result in a file
    result_file = "songlyrics/result_lyrics.txt"
    try:
        os.remove(result_file)
    except OSError:
        pass
    f = open(result_file, 'w')

    # generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        sys.stdout.write(result)
        f.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    f.close()
    print("\nFinished LSTM lyrics generation")