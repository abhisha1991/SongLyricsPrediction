import itertools
import nltk
import numpy as np
import pandas as pd
import random
from SongLyricsPrediction.constants import Constants
# install punkt from nltk models - one time only
# nltk.download()


class LyricsParser(object):
    def __init__(self):
        self.c = Constants()
        self.all_sentences = []
        self.training_sentences = []
        self.tokenized_sentences = []
        self.index_to_word = []
        self.word_to_index = dict()
        self.vocab = []

    def get_sentences(self, mode):
        df = pd.read_csv(self.c.datapath)
        song_lyrics = df['text']
        print("Reading CSV file...")
        sentences = []
        for row in song_lyrics:
            # Each song is fairly long, split by '\n' delimiter to get sentences
            for sent in row.split('\n'):
                sent = sent.strip(' ')
                # Split full song lyrics into sentences
                sentence = ''.join(nltk.sent_tokenize(sent.lower()))
                if mode == self.c.mode_rnn:
                    # Append SENTENCE_START and SENTENCE_END
                    sentences.append(self.c.sentence_start_token + " " + sentence + " " + self.c.sentence_end_token)
                    # print(sentences)
                elif mode == self.c.mode_lstm:
                    sentences.append(sentence)

        return sentences

    def generate_training_data_lstm(self):
        raw_text = ''
        for x in self.training_sentences:
            raw_text = raw_text + ''.join(x) + '.'

        # take the first n characters of the entire corpus
        raw_text = raw_text[:self.c.training_size]

        # create mapping of unique chars to integers
        chars = sorted(list(set(raw_text)))
        char_to_int = dict((ch, i) for i, ch in enumerate(chars))

        # summarize the loaded data
        n_chars = len(raw_text)

        # prepare the data set of input to output pairs encoded as integers
        dataX = []
        dataY = []
        for i in range(0, n_chars - self.c.lstm_seq_length, 1):
            seq_in = raw_text[i:i + self.c.lstm_seq_length]
            seq_out = raw_text[i + self.c.lstm_seq_length]
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])

        return raw_text, dataX, dataY

    def generate_training_data_rnn(self):
        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*self.tokenized_sentences))
        print("Number of unique words found: ", len(word_freq.items()))

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(self.c.vocabulary_size - 1)
        self.index_to_word = [x[0] for x in vocab]
        self.index_to_word.append(self.c.unknown_token)
        self.word_to_index = dict([(w, i) for i, w in enumerate(self.index_to_word)])

        print("Vocabulary size is fixed to ", self.c.vocabulary_size, " words")
        print("The least frequent word in our vocabulary is: ", (vocab[-1][0], vocab[-1][1]))
        print("The most frequent word in our vocabulary is:", (vocab[0][0], vocab[0][1]))

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(self.tokenized_sentences):
            self.tokenized_sentences[i] = [w if w in self.word_to_index else self.c.unknown_token for w in sent]

        '''
        # Create the training data
        for sent in self.tokenized_sentences:
            for w in sent[:-1]:
                print(w)
        '''

        # X_train will contain all the indexes for each word (except SENTENCE_END) in each sentence
        X_train = np.asarray([[self.word_to_index[w] for w in sent[:-1]] for sent in self.tokenized_sentences])

        # y_train will contain all the indexes for each word (except SENTENCE_START) in each sentence
        y_train = np.asarray([[self.word_to_index[w] for w in sent[1:]] for sent in self.tokenized_sentences])

        return X_train, y_train

    def parse_lyrics(self, mode='simple_rnn'):
        self.all_sentences = self.get_sentences(mode)
        print("Number of sentences parsed: ", (len(self.all_sentences)))
        print("Sampling a subset of these, sample size: ", self.c.training_size)

        # There are 2427716 sentences separated by '\n', we are choosing 100,000 random sentences
        self.training_sentences = random.sample(self.all_sentences, self.c.training_size)

        # Tokenize the sentences into words
        for sent in self.training_sentences:
            x = nltk.word_tokenize(sent)
            self.tokenized_sentences.append(x)

        print("\nExample sentence: ", self.training_sentences[0])
        print("\nExample sentence after Pre-processing: ", self.tokenized_sentences[0])

        if mode == self.c.mode_rnn:
            return self.generate_training_data_rnn()

        elif mode == self.c.mode_lstm:
            return self.generate_training_data_lstm()

