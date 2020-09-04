import sys
from utils import *
from baselines import *

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

dataset_name = "reward"

menu_dataset  = MenuDataset(dataset_name + "_price.csv", dataset_name + "_vocab_v2.p", sampling="wp", encoding="utf-8", item_column_name="description", factor_column_name="category", price_column_name="price")

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

# model set up
H = 128
D_out = 1

# training set up
learning_rate = 1e-4
n_epochs = 20

model = wp(menu_dataset.n_vocab, H, D_out)
model.to(device)
model.apply(init_weights) # initialize the weights

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print ("Start training")

for t in range(n_epochs):

    training_loss = 0.0
    validation_loss = 0.0

    n_training_batches  = 0
    n_validation_batches = 0

    # training
    for i_batch, sample_batched in enumerate(train_loader):

        X = sample_batched["words"].to(device)
        y = sample_batched["price"].to(device)

        # forward pass
        y_pred = model(X)

        # backward pass
        loss = criterion(y_pred, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        n_training_batches += 1

    # validation
    with torch.set_grad_enabled(False):
        for i_batch, sample_batched in enumerate(valid_loader):

            X = sample_batched["words"].to(device)
            y = sample_batched["price"].to(device)

            y_pred = model(X)
            loss = criterion(y_pred, y)

            validation_loss += loss.item()
            n_validation_batches += 1

    print ('%d\t%.3f\t%.3f' % (t + 1,  training_loss / n_training_batches, validation_loss / n_validation_batches))
    sys.stdout.flush()

    # save the model at this current epoch
    torch.save(model.state_dict(), "../models/wp_" + dataset_name + "_" + str(t))
