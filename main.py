from SongLyricsPrediction.lyrics_parser import *
from SongLyricsPrediction.models import *
from SongLyricsPrediction.constants import *
from SongLyricsPrediction.utils import *

lp = LyricsParser()
X_train, y_train = lp.parse_lyrics()
c = Constants()


def generate_sentence(model):
    # Start with a start token
    new_sentence = [lp.word_to_index[c.sentence_start_token]]

    # limits to break out of potential infinite loop
    count_inner = 0
    count_outer = 0

    # Repeat until we get an end token
    while not new_sentence[-1] == lp.word_to_index[c.sentence_end_token]:
        o, s = model.forward_propagation(new_sentence)
        sampled_word = lp.word_to_index[c.unknown_token]
        count_outer += 1
        if count_outer == 20:
            new_sentence.append(sampled_word)
            break
        # We don't want to sample unknown words
        while sampled_word == lp.word_to_index[c.unknown_token]:
            samples = np.random.multinomial(1, o[0])
            # Pick the highest probability out of the samples
            sampled_word = np.argmax(samples)
            count_inner += 1
            if count_inner == 20:
                break
        new_sentence.append(sampled_word)
    sentence_str = [lp.index_to_word[x] for x in new_sentence[1:-1]]  # exclude start and end tokens in the final string
    return sentence_str


if __name__ == '__main__':

    # Basic RNN without training on a single observation of the training set
    np.random.seed(10)
    model = RNNNumpy(c.vocabulary_size, c.hidden_layer_size)
    o, s = model.forward_propagation(X_train[10])
    print(len(X_train[10]))
    print(o.shape)
    print(o)
    predictions = model.predict(X_train[10])
    print(predictions.shape)
    print(predictions)

    # A side note, the losses should be of the same order
    print("Expected Loss for random predictions: %f" % np.log(c.vocabulary_size))
    print("Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000]))

    # Basic RNN with training on first n observations
    model = None
    np.random.seed(10)
    model = RNNNumpy(c.vocabulary_size, 50)
    losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)

    num_sentences = 10
    sentence_min_length = 7

    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < sentence_min_length:
            sent = generate_sentence(model)
        print(sent)


