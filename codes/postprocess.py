"""
postprocess.py
"""

from utils import *
from models import *
from baselines import *

# functions
def load_model(menu_dataset_wholeseq, model_name, model_path, hidden_dimension):

    model = None

    if model_name == "joint_wwm":
        model = WWM(1, vocab=menu_dataset_wholeseq.vocab, nb_lstm_units=128, embedding_dim=128, n_factors=menu_dataset_wholeseq.n_factors, batch_size=256)

    elif model_name == "word2vec":
        model = wp(menu_dataset_wholeseq.n_vocab, hidden_dimension, menu_dataset_wholeseq.n_vocab)

    elif model_name == "wp":
        model = wp(menu_dataset_wholeseq.n_vocab, hidden_dimension, 1)

    elif model_name == "mwp":
        model = mwp(menu_dataset_wholeseq.n_vocab, hidden_dimension)

    elif model_name == "mwcp":
        model = mwcp(menu_dataset_wholeseq.n_vocab, menu_dataset_wholeseq.n_factors, hidden_dimension)

    model.load_state_dict(torch.load(model_path))

    return model

def create_dataset(dataset):
    if dataset == "retail":
        menu_fname = "retail_price.csv"
        vocab_fname = "retail_vocab_v2.p"
        encoding = "ISO-8859-1"
        item_column_name = "Description"
        factor_column_name = "Country"
        price_column_name = "UnitPrice"

    elif dataset == "menu":
        menu_fname = "menu_price_1125.csv"
        vocab_fname = "menu_vocab_v2.p"
        encoding = "utf-8"
        item_column_name = "item"
        factor_column_name = "city"
        price_column_name = "price"

    elif dataset == "shoe":
        menu_fname = "shoe_price.csv"
        vocab_fname = "shoe_vocab_v2.p"
        encoding = "ISO-8859-1"
        item_column_name = "name"
        factor_column_name = "brand"
        price_column_name = "prices.amountMax"

    elif dataset == "reward":
        menu_fname = "reward_price.csv"
        vocab_fname = "reward_vocab_v2.p"
        encoding = "utf-8"
        item_column_name = "description"
        factor_column_name = "category"
        price_column_name = "price"

    print ("Create dataset")

    menu_dataset_wholeseq  = MenuDataset(menu_fname, vocab_fname,
                                                                                           sampling="wholeseq", encoding=encoding,
                                                                                           item_column_name=item_column_name,
                                                                                           factor_column_name=factor_column_name,
                                                                                           price_column_name=price_column_name)

    return menu_dataset_wholeseq

def create_loader(menu_dataset):

    validation_split = .2
    train_loader_wholeseq, valid_loader_wholeseq = create_data_loader(menu_dataset, validation_split)

    return train_loader_wholeseq, valid_loader_wholeseq


def extract_embeddings(epoch, dataset, model_name, menu_dataset_wholeseq):

    model_path = "../models/" + "_".join([model_name, dataset, str(epoch)])

    model = load_model(menu_dataset_wholeseq, model_name, model_path, 128)
    model.eval()

    filtered_words = ['<PAD>']
    n_filtered_words = len(filtered_words)
    filtered_words_idx = [menu_dataset_wholeseq.top_words.index(word) for word in filtered_words]
    all_words = [word for word in menu_dataset_wholeseq.top_words if word not in filtered_words]

    if model_name == "joint_wwm":
        X_param = model.word_embedding.weight
        Wc_param = model.Wc
    elif model_name == "word2vec" or model_name == "wp" or model_name == "mwp":
        X_param = model.linear1.weight
        X_param = X_param.transpose(0,1)
    elif model_name == "mwcp":
        X_param = model.linear1_word.weight
        X_param = X_param.transpose(0,1)

    X = {}
    for word in all_words:
        i = menu_dataset_wholeseq.top_words.index(word)
        tmp = []
        for j in range(X_param.shape[1]):
            tmp.append(float(X_param[i][j]))
        X[word] = tmp

    # X = np.asarray(X)

    pickle.dump(X, open("../trained_parameters/"+ model_name + "_" + dataset_name + "_X.p", "wb"))

    if model_name == "joint_wwm" :
        print ("prepare for Wc")
        Wc = {}
        for factor in menu_dataset_wholeseq.all_factors:
            i = menu_dataset_wholeseq.all_factors.index(factor)
            tmp = []
            for j in range(Wc_param.shape[1]):
                tmpp = []
                for k in range(Wc_param.shape[2]):
                    tmpp.append(float(Wc_param[i][j][k]))
                tmp.append(tmpp)
            Wc[factor] = tmp

        # Wc = np.asarray(Wc)

        pickle.dump(Wc, open("../trained_parameters/"+ model_name + "_" + dataset_name + "_Wc.p", "wb"))


model_names = ["joint_wwm"]
dataset_names = ["shoe"]

for dataset_name in dataset_names:
    print ("Dataset:", dataset_name)

    menu_dataset_wholeseq = create_dataset(dataset_name)

    for model_name in model_names:
        print ("Model:", model_name)

        if model_name == "joint_wwm":
            epoch = 99
        else:
            epoch = 19

        extract_embeddings(epoch, dataset_name, model_name, menu_dataset_wholeseq)
