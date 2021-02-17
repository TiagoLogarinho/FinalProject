#Imports
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
###########################################

data = pd.read_csv('Datasets/archive/bitcoin_price.csv')
x = data.drop(['Close','Date','Volume','Market Cap'], axis = 1).values[::-1]
y = data['Close'].values[::-1]
#linear regression
def linear_regression(x,y):
    #scaler = MinMaxScaler(feature_range=(0, 1)).fit_transform(x,y)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    scores = cross_val_score(model, x_test, y_test, cv=5, scoring='accuracy')
    print("Training score: {:.3f}".format(model.score(x_train, y_train)))
    print("Testing score: {:.3f}".format(model.score(x_test, y_test)))
    print("Model Accuracy: {:.3f}".format(model.score(y_test, y_test_pred)))
    print("Cross Val Score: {:.3}".format(scores.mean()))
    print("Actual Test labels: {0}".format(y_test[:5]))
    print("Predicted labels: {0}".format(y_test_pred[:5]))
    #plt.figure(figsize=(16,8))
    #plt.plot(y_test)
    #plt.plot(y_test_pred[::-1])
    #plt.show()
    #for x, y in zip(y_test, y_test_pred):
    #    print("Actual: {:}".format(x), "Test pred: {:}".format(y))
    #new_y = y[len(y_train_pred):,]
    #print(new_y.shape)
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

#random forests
def random_forests(x,y):
    scaler = MinMaxScaler(feature_range=(0, 1)).fit_transform(x,y)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=42)
    model = RandomForestRegressor().fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("Training score: {:.3f}".format(model.score(x_train, y_train)))
    print("Testing score: {:.3f}".format(model.score(x_test, y_test)))
    print("Actual Test labels: {0}".format(y_test[:5]))
    print("Predicted labels: {0}".format(y_test_pred[:5]))
    #y.shape = 1760, y_train_pred.shape = 1232, y_test_pred = 528
    """train_predict_plot = np.empty_like(y)
    test_predict_plot = np.empty_like(y)
    train_predict_plot[:,] = np.nan 
    test_predict_plot[:,] = np.nan
    train_predict_plot[0 : len(y_train_pred)] = y_train_pred
    test_predict_plot[len(y_train_pred):,] = y_test_pred
    plt.figure(figsize=(16,8))
    plt.plot(train_predict_plot, 'r')
    plt.plot(y, 'g')
    plt.plot(test_predict_plot, 'b')
    plt.show()"""
    #for x, y in zip(y[len(y_train_pred):,], y_test_pred):
    #    print("Actual: {:}".format(x), "Test pred: {:}".format(y))
    new_y = y[len(y_train_pred):,]
    print(new_y.shape)
#random_forests(x,y)
#TODO: Try and create a model that collects data on daily basis and make it recurring.
def progressive_prediction(x,y):
    pass