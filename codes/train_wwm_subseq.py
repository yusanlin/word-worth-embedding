import sys
from utils import *
from models import *

from torch.utils.data.sampler import SubsetRandomSampler

# GPU information
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)

# settings
validation_split = .2

print ("Construct dataset and loader")

menu_dataset  = MenuDataset("menu_price_1125.csv", "menu_vocab.p", sampling="subseq", beta=.5)

# split into train and valid sets
dataset_size = len(menu_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# create samplers for training and validation
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

# create data loaders for training and validation
train_loader = DataLoader(menu_dataset, batch_size=256, sampler=train_sampler)
valid_loader = DataLoader(menu_dataset, batch_size=256, sampler=valid_sampler)

# training set up
learning_rate = 1e-3
n_epochs = 20


model = WWM(1, vocab=menu_dataset.vocab, nb_lstm_units=256, embedding_dim=256, n_factors=menu_dataset.n_factors, batch_size=256)
model.to(torch.device("cuda:0"))


"""
Reference: https://discuss.pytorch.org/t/multiclass-classification-with-nn-crossentropyloss/16370
"""
criterion_word = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print ("Start training")

for t in range(n_epochs):

    training_loss = 0.0
    validation_loss = 0.0

    n_training_batches  = 0
    n_validation_batches = 0

    # training
    for i_batch, sample_batched in enumerate(train_loader):

        # if i_batch % 100 == 0: print ("i_batch:", i_batch)

        X_subseq = sample_batched["subseq"]
        X_missing = sample_batched["missing"]
        Xc = sample_batched["factor"]
        y = sample_batched["price"]

        X_lengths = sample_batched["length"]

        lengths_sorted, sorted_idx = X_lengths.sort(descending=True)
        X_subseq = X_subseq[sorted_idx]
        X_missing = X_missing[sorted_idx]
        Xc = Xc[sorted_idx]
        y = y[sorted_idx]

        lengths_sorted = lengths_sorted.to(torch.device("cuda:0"))
        X_subseq = X_subseq.to(torch.device("cuda:0"))
        X_missing = X_missing.to(torch.device("cuda:0"))
        Xc = Xc.to(torch.device("cuda:0"))
        y = y.to(torch.device("cuda:0"))

        # forward pass
        try:
            output_m, output_y =  model(X_subseq, lengths_sorted, Xc)

            # backward pass
            # input: (minibatch, N_CLASSES, SEQ_LEN)
            # target: (minibatch, SEQ_LEN)
            # TODO: 08/30 https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
            loss = criterion_word(output_m, X_missing.squeeze())
            model.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            n_training_batches += 1
        except RuntimeError:
            pass

    # validation
    with torch.set_grad_enabled(False):
        for i_batch, sample_batched in enumerate(valid_loader):

            X_subseq = sample_batched["subseq"]
            X_missing = sample_batched["missing"]
            Xc = sample_batched["factor"]
            y = sample_batched["price"]

            X_lengths = sample_batched["length"]

            lengths_sorted, sorted_idx = X_lengths.sort(descending=True)
            X_subseq = X_subseq[sorted_idx]
            X_missing = X_missing[sorted_idx]
            Xc = Xc[sorted_idx]
            y = y[sorted_idx]

            lengths_sorted = lengths_sorted.to(torch.device("cuda:0"))
            X_subseq = X_subseq.to(torch.device("cuda:0"))
            X_missing = X_missing.to(torch.device("cuda:0"))
            Xc = Xc.to(torch.device("cuda:0"))
            y = y.to(torch.device("cuda:0"))

            try:
                output_m, output_y =  model(X_subseq, lengths_sorted, Xc)
                loss = criterion_word(output_m, X_missing.squeeze())

                validation_loss += loss.item()
                n_validation_batches += 1
            except RuntimeError:
                pass

    print ('%d\t%.3f\t%.3f' % (t + 1,  training_loss / n_training_batches, validation_loss / n_validation_batches))
    sys.stdout.flush()

    # save the model at this current epoch
    torch.save(model.state_dict(), "../models/wwm_subseq_" + str(t))
