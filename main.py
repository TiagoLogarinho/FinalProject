#Imports
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle
###########################################

data = pd.read_csv('Datasets/archive/bitcoin_price.csv')
x = data.drop(['Close','Date','Volume','Market Cap'], axis = 1).values
y = data['Close'].values
#plt.figure(figsize=(16,8))
#plt.plot(y[::-1])
#plt.show()
#linear regression
def linear_regression(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=0)
    model = LinearRegression()
    model.fit(x_train, y_train)
    #y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("Training score: {:.3f}".format(model.score(x_train, y_train)))
    print("Testing score: {:.3f}".format(model.score(x_test, y_test)))
    print("Actual Test labels: {0}".format(y_test[:5]))
    print("Predicted labels: {0}".format(y_test_pred[:5]))
    plt.figure(figsize=(16,8))
    #plt.plot(y_test)
    plt.plot(y_test_pred[::-1])
    plt.show()
linear_regression(x,y)
#TODO: More models (NN, RF, KNN) and tuned versions of said models

#neural networks
def neural_networks(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=0)
    model = MLPRegressor()
    model.fit(x_train, y_train)
    #y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("Training score: {:.3f}".format(model.score(x_train, y_train)))
    print("Testing score: {:.3f}".format(model.score(x_test, y_test)))
    print("Actual Test labels: {0}".format(y_test[:5]))
    print("Predicted labels: {0}".format(y_test_pred[:5]))
#neural_networks(x,y)
#TODO: Try and create a model that collects data on daily basis and make it recurring.
def progressive_prediction(x,y):
    pass