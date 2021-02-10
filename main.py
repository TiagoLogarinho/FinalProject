#Imports
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle
###########################################

data = pd.read_csv('Datasets/archive/bitcoin_price.csv', )

x = data.drop(['Close','Date','Volume','Market Cap'], axis = 1).values
y = data['Close'].values
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=0)
model = LinearRegression()
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
print("Training score: {:.3f}".format(model.score(x_train, y_train)))
print("Testing score: {:.3f}".format(model.score(x_test, y_test)))
print("Actual Test labels: {0}".format(y_test[:5]))
print("Predicted labels: {0}".format(y_test_pred[:5]))
plt.plot(y_test)
plt.plot(y_test_pred)
plt.show()