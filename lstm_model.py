import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from SongLyricsPrediction.constants import *

c = Constants()


def lstm_lyrics_generator(X_train):
    # load ascii text
    raw_text = ''
    for x in X_train:
        raw_text = raw_text + ''.join(x) + '.'

    raw_text = raw_text[:1000000]
    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((ch, i) for i, ch in enumerate(chars))
    int_to_char = dict((i, ch) for i, ch in enumerate(chars))

    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)

    # prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []
    for i in range(0, n_chars - c.lstm_seq_length, 1):
        seq_in = raw_text[i:i + c.lstm_seq_length]
        seq_out = raw_text[i + c.lstm_seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, c.lstm_seq_length, 1))

    # normalize
    X = X / float(n_vocab)

    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # define the LSTM model
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
    resultfile = "songlyrics/result_lyrics.txt"
    try:
        numpy.os.remove(resultfile)
    except OSError:
        pass
    f = open(resultfile, 'w')

    # generate characters
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        f.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    f.close()
print("\nFinished LSTM lyrics generation")