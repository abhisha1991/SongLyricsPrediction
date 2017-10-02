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

We used LSTMs on a character level prediction model using LSTMs. The results have been interesting - from some of the output files we can see that the LSTM starts generating some lyrics which make sense and it comes close to predicting words like "know", "that", "you're". It tries a couple of original lines, but then gets stuck in a loop and spits out the same sentences, such as "i'm sorry koow so say your heart..." - the best guess would be that these lyrics were a part of some chorus of a song and were repeated a couple of times, so the RNN tries to mimic chorus lyrics. 

It is also important to note that the model was trained on only some 100,000 characters from a random set of song lyrics of different genres. Hence, it may not make much sense as the training data itself was not very uniform in its tone/meaning. The solution would be to find abundant lyrics specific to a band/genres which speak about common themes in songs. Finally, experimentation with the hyperparameters needs to be performed to generate different models that would work better. However, training one of these models on a standard CPU VM (4 cores, 14 GB RAM, 2.4 Ghz) takes upto 16 hours (2 layer LSTM with 256 layers each). Hence, significant compute/GPUs/FPGAs would need to be employed to generate multiple models to make the training time feasible. 

## Dependencies
1. nltk
2. numpy
3. pandas
4. install punkt from nltk models - one time only

## Sample usage
Simply clone the repo and run **python main.py**

## Credits
The credits of this hugely go to the tutorial that was made by Denny Britz - https://github.com/dennybritz/rnn-tutorial-rnnlm and his blog
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
