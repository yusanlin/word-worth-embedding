import torch

class wp(torch.nn.Module):
    """
    This can be used for word2vec as well, when D_out != 1
    """
    def __init__(self, D_in, H, D_out):

        super(wp, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.b_norm = torch.nn.BatchNorm1d(H)
        self.l_norm = torch.nn.LayerNorm(H)

    def forward(self, x):

        h_relu = self.linear1(x).clamp(min=0)
        h_relu = self.b_norm(h_relu)
        h_relu = self.l_norm(h_relu)
        y_pred = self.linear2(h_relu)
        return y_pred

class mwp(torch.nn.Module):
    def __init__(self, D_in, H):

        super(mwp, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_in)
        self.linear3 = torch.nn.Linear(H, 1)
        self.b_norm = torch.nn.BatchNorm1d(H)
        self.l_norm = torch.nn.LayerNorm(H)

    def forward(self, x):

        h_relu = self.linear1(x).clamp(min=0)
        h_relu = self.b_norm(h_relu)
        h_relu = self.l_norm(h_relu)
        x_pred = self.linear2(h_relu)
        y_pred = self.linear3(h_relu)

        return x_pred, y_pred

class mwcp(torch.nn.Module):
    def __init__(self, D_in_word, D_in_factor, H):

        super(mwcp, self).__init__()
        self.linear1_word = torch.nn.Linear(D_in_word, H)
        self.linear1_factor = torch.nn.Linear(D_in_factor, H)
        self.linear2 = torch.nn.Linear(H, D_in_word)
        self.linear3 = torch.nn.Linear(H, 1)
        self.b_norm = torch.nn.BatchNorm1d(H)
        self.l_norm = torch.nn.LayerNorm(H)

    def forward(self, x_word, x_factor):

        h_word = self.linear1_word(x_word)
        h_factor = self.linear1_factor(x_factor)
        h_relu = h_word.add(h_factor).clamp(min=0)
        h_relu = self.b_norm(h_relu)
        h_relu = self.l_norm(h_relu)
        x_pred = self.linear2(h_relu)
        y_pred = self.linear3(h_relu)

        return x_pred, y_pred



# Functions to initialize weights in models
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
