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
        self.tokenized_sentences = []
        self.index_to_word = []
        self.word_to_index = dict()
        self.vocab = []

    def get_sentences(self):
        df = pd.read_csv(self.c.datapath)
        song_lyrics = df['text']
        print("Reading CSV file...")
        sentences = []
        for row in song_lyrics:
            # Each song is fairly long, split by '\n' delimiter to get sentences
            for sent in row.split('\n'):
                sent = sent.strip(' ')
                # Split full song lyrics into sentences
                sentence = ''.join(nltk.sent_tokenize(sent))
                # Append SENTENCE_START and SENTENCE_END
                sentences.append("%s %s %s" % (self.c.sentence_start_token, sentence, self.c.sentence_end_token))
                # print(sentences)
        return sentences

    def parse_lyrics(self):
        self.all_sentences = self.get_sentences()
        # There are 2427716 sentences separated by '\n', we are choosing 100,000 random sentences
        training_sentences = random.sample(self.all_sentences, self.c.training_size)
        print("Number of sentences parsed: ", (len(self.all_sentences)))

        # Tokenize the sentences into words
        for sent in training_sentences:
            x = nltk.word_tokenize(sent)
            self.tokenized_sentences.append(x)

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
        print("The most frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[0][0], vocab[0][1]))

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(self.tokenized_sentences):
            self.tokenized_sentences[i] = [w if w in self.word_to_index else self.c.unknown_token for w in sent]

        print("\nExample sentence: '%s'" % training_sentences[0])
        print("\nExample sentence after Pre-processing: '%s'" % self.tokenized_sentences[0])

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


