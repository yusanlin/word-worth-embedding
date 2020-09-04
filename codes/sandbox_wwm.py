import torch
import numpy as np

from torch.autograd import Variable
from torch.nn import functional as F

# Let's start from inputting data into a simple RNN first

class TutorialLSTM(torch.nn.Module):

    def __init__(self, nb_layers, nb_lstm_units=100, nb_lstm_layers=1, embedding_dim=3, n_factors=3, batch_size=3):

        super(TutorialLSTM, self).__init__()

        self.vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8, 'yeah':9}
        self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}

        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.nb_lstm_layers = nb_lstm_layers
        self.embedding_dim = embedding_dim
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.nb_vocab_words = len(self.vocab)

        self.nb_tags = len(self.tags) - 1

        # self.lstm

        self.__build_model()

    def __build_model(self):

        nb_vocab_words = len(self.vocab)

        padding_idx = self.vocab['<PAD>']
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=nb_vocab_words,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx
        )

        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )

        self.hidden_to_tag = torch.nn.Linear(self.nb_lstm_units, self.nb_tags)

        # the feedforward NN for the attention layer
        # self.attetion_a = torch.nn.Linear()

        # the feedforward NN for the contextual
        # TODO: change to Xavier initialization later
        self.Wc = torch.randn(self.n_factors, self.nb_lstm_units, self.nb_lstm_units)
        self.Wm = torch.randn(self.nb_lstm_units, self.nb_vocab_words)
        self.Wy = torch.randn(self.nb_lstm_units, 1)

    def init_hidden(self):
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return  (hidden_a, hidden_b)

    def forward(self, X, X_lengths, Xc):
        self.hidden = self.init_hidden()
        # self.hidden = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        batch_size, seq_len = X.size()

        X = self.word_embedding(X)

        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        X, self.hidden = self.lstm(X, self.hidden)

        hs, _ = self.hidden

        # print ("hs:", hs) #sequence embeedding

        # TODO: attention layer here

        bc = torch.matmul(hs.transpose(0,1), self.Wc[torch.argmax(Xc, axis=0)])
        bc = torch.squeeze(bc)

        # print ("hs + bc:", (hs+bc).shape)
        # print ("(hs + bc) * Wm)", torch.matmul(hs + bc, self.Wm).shape)

        output_m = torch.matmul(hs + bc, self.Wm)
        output_y = torch.matmul(hs + bc, self.Wy)
        #
        #
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        #
        # X = X.contiguous()
        # X = X.view(-1, X.shape[2])
        #
        # X = self.hidden_to_tag(X)
        #
        # X = F.log_softmax(X, dim=1)
        #
        # X = X.view(batch_size, seq_len, self.nb_tags)
        #
        # y_hat = X
        #
        # return y_hat
        return output_m, output_y

    def _sequence_mask(self, sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()

        batch_size = sequence_length.size(0)
        seq_range = torch.range(0, max_len - 1).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)

        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))

        return seq_range_expand < seq_length_expand

    def compute_loss(self, Y_hat, Y, X_lenghts):
        """
        Reference: https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
        """

        logits_flat = Y_hat.view(-1, Y_hat.size(-1))
        log_probs_flat = F.log_softmax(logits_flat)

        target_flat = Y.view(-1, 1)
        target_flat -= 1
        target_flat[target_flat < 0] = 0

        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

        losses = losses_flat.view(*Y.size())

        mask = self._sequence_mask(sequence_length=X_lengths, max_len=Y.size(1))
        losses = losses * mask.float()
        loss = losses.sum() / X_lengths.float().sum()

        return loss

sent_1_x = ['is', 'it', 'too', 'late', 'now', 'say', 'sorry']
sent_1_y = ['VB', 'PRP', 'RB', 'RB', 'RB', 'VB', 'JJ']

sent_2_x = ['ooh', 'ooh']
sent_2_y = ['NNP', 'NNP']

sent_3_x = ['sorry', 'yeah']
sent_3_y = ['JJ', 'NNP']

X = [sent_1_x, sent_2_x, sent_3_x]
Y = [sent_1_y, sent_2_y, sent_3_y]

vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8, 'yeah':9}
tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}

X = [[vocab[word] for word in sentence] for sentence in X]
Y = [[tags[tag] for tag in sentence] for sentence in Y]

# pad X
X_lengths = [len(sentence) for sentence in X]


pad_token = vocab['<PAD>']
longest_sent = max(X_lengths)
batch_size  = len(X)
padded_X = np.ones((batch_size, longest_sent)) * pad_token

for i, x_len in enumerate(X_lengths):
    sequence = X[i]
    padded_X[i, 0:x_len] = sequence[:x_len]

# pad Y
Y_lengths = [len(sentence) for sentence in Y]

pad_token = tags['<PAD>']
longest_sent = max(Y_lengths)
batch_size = len(Y)
padded_Y = np.ones((batch_size, longest_sent)) * pad_token

for i, y_len in enumerate(Y_lengths):
    sequence = Y[i]
    padded_Y[i, 0:y_len] = sequence[:y_len]

padded_X = np.asarray(padded_X)
X_lengths = np.asarray(X_lengths)

padded_Y = np.asarray(padded_Y)
Y_lengths = np.asarray(Y_lengths)

Xc  = np.zeros((3, 3))
Xc[0,0] = 1
Xc[1,1] = 1
Xc[2,2] = 1

# https://github.com/huggingface/transformers/issues/2952
padded_X = torch.from_numpy(padded_X).type(torch.LongTensor)
X_lengths = torch.from_numpy(X_lengths)

padded_Y = torch.from_numpy(padded_Y).type(torch.LongTensor)
Y_lengths = torch.from_numpy(Y_lengths)

Xc = torch.from_numpy(Xc).type(torch.float)

model = TutorialLSTM(1)
output_m, output_y = model(padded_X, X_lengths, Xc)
# loss = model.compute_loss(y_pred, padded_Y, X_lengths)
