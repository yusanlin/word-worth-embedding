"""
create_vocab.py
@date: 2020/08/31
"""

import string
import pickle
import enchant
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter

all_stopwords = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()
english_dict = enchant.Dict("en_US")

def remove_digits(s):
    return ''.join([i for i in s if not i.isdigit()])

def lemmatize(s):
    return wordnet_lemmatizer.lemmatize(s.lower().encode("ascii", "ignore").decode("utf-8"))

data_path = "../../data/"
menu_fname = "shoe_price.csv"
menu_frame = pd.read_csv(data_path + menu_fname, na_values="None", encoding="ISO-8859-1")

words = []

for item in menu_frame["name"]:
    try:
        tmp_words = item.lower().split(" ")
        tmp_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in tmp_words]
        tmp_words = [remove_digits(word) for word in tmp_words]
        tmp_words = [lemmatize(word) for word in tmp_words]
        tmp_words = [word for word in tmp_words if len(word) > 1]
        tmp_words = [word for word in tmp_words if english_dict.check(word)]
        tmp_words = [word.strip() for word in tmp_words]
        # tmp_words = [word for word in all_stopwords if word not in all_stopwords]
        words.extend(tmp_words)
    except AttributeError:
        pass

word_counter = Counter(words)

rank = 0
i = 0

word_counter_populated = word_counter.most_common()

vocab = []

while rank < 500:
    word, count = word_counter_populated[i]
    if word not in all_stopwords and word != "":
        print (word, count)
        vocab.append(word)
        rank += 1
    i += 1

pickle.dump(vocab, open("shoe_vocab_v2.p", "wb"))
