#Imports
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
###########################################

data = pd.read_csv('Datasets/archive/bitcoin_price.csv', )
print(data.head)
