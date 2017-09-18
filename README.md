## Background

This project attempts to predict the next sequence of words (lyrics) based on a given set of song lyrics. Currently, the data set of song lyrics being used is from Kaggle (https://www.kaggle.com/mousehead/songlyrics) - It contains 55k songs with bands ranging from ABBA to Queen to Metallica.

## Process

We first tokenize the song lyrics into sentences and words and make a mapping of the most frequent words used in the corpus. The word to index mapping maps the word in the 'vocabulary' to a given index. We use a vocabulary of 8000 words, however that is configurable.
We break down a sentence into tokenized words such that a sentence always begins with a start token and ends with an end token.
For example **START Remember when you were young, you shone like the sun, shine on you crazy diamond END**

The y values are the next values of words in the sample setence.

We first attempt to predict 10 (configurable) sentences of some words each (we try to predict longer sentences, so we keep sentence length atleast 7 (configurable)).
This is done using a basic RNN trained on SGD. The results are not impressive as RNNs cannot derive context from the past too well.
The next attempt is to use LSTMs.

