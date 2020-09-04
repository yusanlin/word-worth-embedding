import torch
import numpy as np

from torch.autograd import Variable
from torch.nn import functional as F

class WWM(torch.nn.Module):

    def __init__(self, nb_layers, vocab, nb_lstm_units=100, nb_lstm_layers=1, embedding_dim=3, n_factors=3, batch_size=3, dropout_rate=0.2, max_word_length=10):

        super(WWM, self).__init__()

        # self.vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8, 'yeah':9}
        # self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}

        self.vocab = vocab

        self.nb_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.nb_lstm_layers = nb_lstm_layers
        self.embedding_dim = embedding_dim
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.nb_vocab_words = len(self.vocab)

        self.max_word_length = max_word_length

        self.dropout_rate = dropout_rate

        # self.nb_tags = len(self.tags) - 1

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

        self.rnn = torch.nn.RNN(
            input_size = self.embedding_dim,
            hidden_size = self.nb_lstm_units,
            num_layers = self.nb_lstm_layers,
            batch_first = True,
        )

        self.drop_out = torch.nn.Dropout(p=self.dropout_rate)

        # self.hidden_to_tag = torch.nn.Linear(self.nb_lstm_units, self.nb_tags)

        # the feedforward NN for the attention layer
        # self.attetion_a = torch.nn.Linear()

        # the feedforward NN for the contextual
        # TODO: change to Xavier initialization later
        self.Wr = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units).to(torch.device("cuda:0"))
        self.Wc = torch.randn(self.n_factors, self.nb_lstm_units, self.nb_lstm_units).to(torch.device("cuda:0"))
        # self.Wm = torch.randn(self.nb_lstm_units, self.nb_vocab_words).to(torch.device("cuda:0"))
        self.Wm = torch.nn.Linear(self.nb_lstm_units, self.nb_vocab_words)
        # self.Wy = torch.randn(self.nb_lstm_units, 1).to(torch.device("cuda:0"))
        self.Wy = torch.nn.Linear(self.nb_lstm_units, 1)

        self.Wr = torch.nn.init.xavier_normal_(self.Wr, gain=1.0)
        self.Wc = torch.nn.init.xavier_normal_(self.Wc, gain=1.0)
        # self.Wm = torch.nn.init.xavier_normal_(self.Wm, gain=1.0)
        # self.Wy = torch.nn.init.xavier_normal_(self.Wy, gain=1.0)


    def init_hidden(self):
        # hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        # hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        hidden_a = torch.randn(self.nb_lstm_layers, self.embedding_dim, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_lstm_layers, self.embedding_dim, self.nb_lstm_units)

        hidden_a = torch.nn.init.xavier_normal_(hidden_a, gain=1.0)
        hidden_b = torch.nn.init.xavier_normal_(hidden_b, gain=1.0)

        hidden_a = Variable(hidden_a).to(torch.device("cuda:0"))
        hidden_b = Variable(hidden_b).to(torch.device("cuda:0"))

        return  (hidden_a, hidden_b)

    def forward(self, X, X_lengths, Xc):
        self.hidden = self.init_hidden()

        batch_size, seq_len = X.size()

        X = self.word_embedding(X)
        length = X.size(0)
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # print ("X.data.shape:", X.data.shape)

        # X, self.hidden = self.lstm(X, self.hidden)
        # hs, _ = self.hidden

        X, hs = self.rnn(X, self.Wr)

        # TODO: attention layer here

        bc = torch.matmul(hs.transpose(0,1), self.Wc[torch.argmax(Xc, axis=1)])
        bc = torch.squeeze(bc)

        bc = self.drop_out(bc)

        output_m = self.Wm(hs + bc)
        # output_y = torch.tanh(torch.matmul(hs + bc, self.Wy))
        output_y = self.Wy(hs + bc)

        output_m = output_m.squeeze()
        # output_m.unsqueeze_(-1)
        # output_m = output_m.expand(output_m.size(0), self.nb_vocab_words, self.max_word_length)
        # output_m = output_m.transpose(-1, -2)

        output_m = self.drop_out(output_m)
        output_y = self.drop_out(output_y)

        return output_m, output_y
