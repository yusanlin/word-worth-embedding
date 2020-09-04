"""
create_factors.py
@date: 2020/09/02
"""

import pickle
import pandas as pd

data_path = "../../data/"
menu_fname = "retail_price.csv"
encoding = "ISO-8859-1" # "ISO-8859-1", "utf-8"
menu_frame = pd.read_csv(data_path + menu_fname, na_values="None", encoding=encoding)
factor_column_name = "Country"

factors = []

for factor in menu_frame[factor_column_name]:
    factors.append(str(factor))

factors = list(set(factors))
factors.sort()

print (len(factors), "factors in total.")

pickle.dump(factors, open("retail_factors.p", "wb"))
