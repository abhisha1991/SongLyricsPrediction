from SongLyricsPrediction.utils import *


class RNNNumpy:
    def __init__(self, word_dim, hidden_dim, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim  # size of vocab
        self.hidden_dim = hidden_dim  # size of hidden layer
        self.bptt_truncate = bptt_truncate

        # U = incoming weights -- dim = hidden_dim x word_dim
        # V = outgoing weights -- dim = word_dim x hidden_dim
        # W = outgoing weights which feed back into the hidden layer -- dim == hidden_dim x hidden_dim
        # St = output at time t from hidden unit -- dim = hidden_dim x 1
        # Xt = input at time t (= St-1) -- dim = word_dim x 1
        # Ot = output of that layer --dim = word_dim x 1

        # St = tanh(U.Xt + W.St-1)
        # Ot = softmax(V.St)

        # initialize weights randomly between -1/sqrt(n) and 1/sqrt(n)
        # why? https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, xt):
        # The total number of time steps
        T = len(xt)
        # During forward propagation we save all hidden states in s because need them later.
        # timesteps x hidden_dim initialization to 0
        s = np.zeros((T + 1, self.hidden_dim))
        # Add one additional element for the initial hidden, which we set to 0
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:, xt[t]] + self.W.dot(s[t - 1]))
            # same as s[t] = np.tanh(self.U.dot(xt[t]) + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # Return index of the max probability value along the rows of output, which has dim = word_dim x 1
        # This will give the index of the next predicted word
        # Here x is a sentence of 'n' words (say)
        # Then x's dimension itself will contain some n words, hence o's dimension becomes n x word_dim, which means
        # instead of getting a single word prediction, we get an n word prediction
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0 # initialize loss to 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]


