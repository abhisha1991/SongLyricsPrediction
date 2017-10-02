from SongLyricsPrediction.lyrics_parser import *
from SongLyricsPrediction.lstm_model import *
import numpy
from keras.models import Sequential
import os
import sys

# Contains very similar contents compared to the lstm_model file,
# use only when you already have a model and you want to generate lyrics

c = Constants()
lp = LyricsParser()

print("Starting LSTM lyrics generation")
raw_text, dataX, dataY = lp.parse_lyrics(c.mode_lstm)

chars = sorted(list(set(raw_text)))
n_chars = len(raw_text)
n_vocab = len(chars)
n_patterns = len(dataX)
int_to_char = dict((i, ch) for i, ch in enumerate(chars))
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, c.lstm_seq_length, 1))

# normalize
X = X / float(n_vocab)

# one hot encode the output variable
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

filepath = "songlyrics/lstm_saved.hdf5"

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