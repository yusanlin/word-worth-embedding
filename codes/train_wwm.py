import sys
from utils import *
from models import *

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

# construct one dataset for whole sequence sampling, and one for subsequence sampling
menu_dataset_wholeseq  = MenuDataset(dataset_name + "_price.csv", dataset_name + "_vocab_v2.p", sampling="wholeseq", encoding="utf-8", item_column_name="description", factor_column_name="category", price_column_name="price")
menu_dataset_subseq = MenuDataset(dataset_name + "_price.csv", dataset_name + "_vocab_v2.p", sampling="subseq", beta=.5, encoding="utf-8", item_column_name="description", factor_column_name="category", price_column_name="price")

train_loader_wholeseq, valid_loader_wholeseq = create_data_loader(menu_dataset_wholeseq, validation_split)
train_loader_subseq, valid_loader_subseq = create_data_loader(menu_dataset_subseq, validation_split)

# training set up
alpha = .5
learning_rate = 1e-3
n_epochs = 100

model = WWM(1, vocab=menu_dataset_wholeseq.vocab, nb_lstm_units=128, embedding_dim=128, n_factors=menu_dataset_wholeseq.n_factors, batch_size=256)
model.to(device)

criterion_price = torch.nn.MSELoss(reduction='sum')
criterion_word = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print ("Start training")

for t in range(n_epochs):

    training_loss_price, validation_loss_price = 0.0, 0.0
    training_loss_word, validation_loss_word = 0.0, 0.0

    n_training_batches_price, n_validation_batches_price  = 0, 0
    n_training_batches_word, n_validation_batches_word  = 0, 0

    sampling = flip(alpha)

    if sampling == "wholeseq":
        train_loader = train_loader_wholeseq
        valid_loader = valid_loader_wholeseq
    else:
        train_loader = train_loader_subseq
        valid_loader = valid_loader_subseq

    # training
    for i_batch, sample_batched in enumerate(train_loader):

        # if i_batch % 100 == 0: print ("i_batch:", i_batch)

        data = process_batch_to_input(sample_batched, sampling, device)

        try:

            if sampling == "wholeseq":
                _, output_y =  model(data["X"], data["lengths"], data["Xc"])
                loss = criterion_price(output_y,  data["y"])
            else:
                output_m, output_y =  model(data["X"],data["lengths"], data["Xc"])
                loss = criterion_word(output_m, data["y_word"])

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if sampling == "wholeseq":
                training_loss_price += loss.item()
                n_training_batches_price += 1
            else:
                training_loss_word += loss.item()
                n_training_batches_word += 1
        except RuntimeError:
            pass

    # validation
    with torch.set_grad_enabled(False):
        for i_batch, sample_batched in enumerate(valid_loader):

            data = process_batch_to_input(sample_batched, sampling, device)

            try:
                if sampling == "wholeseq":
                    _, output_y =  model(data["X"], data["lengths"], data["Xc"])
                    loss = criterion_price(output_y,  data["y"])

                    validation_loss_price += loss.item()
                    n_validation_batches_price += 1
                else:
                    output_m, output_y =  model(data["X"],data["lengths"], data["Xc"])
                    loss = criterion_word(output_m, data["y_word"])

                    validation_loss_word += loss.item()
                    n_validation_batches_word += 1
            except RuntimeError:
                pass

    if sampling == "wholeseq":
        print ('wholeseq %d\t%.3f\t%.3f' % (t + 1,  training_loss_price / n_training_batches_price, validation_loss_price / n_validation_batches_price))
    else:
        print ('subseq %d\t%.3f\t%.3f' % (t + 1,  training_loss_word / n_training_batches_word, validation_loss_word / n_validation_batches_word))
    sys.stdout.flush()

    # save the model at this current epoch
    torch.save(model.state_dict(), "../models/joint_wwm_" + dataset_name + "_" + str(t))
