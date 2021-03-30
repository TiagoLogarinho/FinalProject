#Imports
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math
import requests
from time import time, sleep
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from matplotlib.legend_handler import HandlerLine2D
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor, PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, train_test_split,  GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import max_error, explained_variance_score, r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
###########################################

def create_datasets(dataset, sequence_length):
    sequence_length += 1
    seq_dataset = []
    for i in range(len(dataset) - sequence_length):
        seq_dataset.append(dataset[i: i + sequence_length])
    seq_dataset = np.array(seq_dataset)

    data_x = seq_dataset[:, :-1]
    data_y = seq_dataset[:, -1]
    
    return data_x, data_y

def plot_data(y, train, test, labels, title):
    plt.figure(figsize=(14,6))
    plt.plot(y, 'b', label=labels[0])
    plt.plot(train, 'g', label=labels[1])
    plt.plot(test, 'r', label=labels[2])
    plt.xlabel('Time in days')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()

def split_data(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 10
    test_size = int(len(prices)*0.2)
    prices['Prediction'] = prices[['Close']].shift(-prediction_range)
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close','Prediction'], axis=1))
    x = x[:len(prices) - prediction_range]
    y = np.array(prices['Prediction'])
    y = y[: -prediction_range]
    #Scale the data
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    #Split data into training and testing sets
    x_train, x_test = x_scaled[:len(prices) - prediction_range - test_size], x_scaled[len(prices) - prediction_range - test_size:]
    y_train, y_test = y[:len(prices) - prediction_range - test_size], y[len(prices) - prediction_range - test_size:]
    return x, y, x_train, x_test, y_train, y_test

def prepare_plots(y, y_train, y_train_pred, y_test_pred):
    new_y_test_pred = np.empty_like(y)
    new_y_train_pred = np.empty_like(y)
    new_y_test_pred[:] = np.nan
    new_y_train_pred[:] = np.nan
    new_y_test_pred[len(y_train):] = y_test_pred
    new_y_train_pred[:len(y_train)] = y_train_pred
    return new_y_train_pred, new_y_test_pred

#linear regression
def linear_regression_btc(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = LinearRegression()
    #Tune model
    model_params = {
        'fit_intercept': ['True', 'False'],
        'normalize': ['True', 'False'],
        'n_jobs': [-1, 1]
    }
    tuned_model = GridSearchCV(model, model_params, cv=5, scoring='r2')
    tuned_model.fit(x_train, y_train)
    y_test_pred = tuned_model.predict(x_test)
    y_train_pred = tuned_model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual BTC Prices','Predicted Training Prices','Predicted Prices'],'LR BTC Price Prediction')

def linear_regression_eth(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = LinearRegression()
    #Tune model
    model_params = {
        'fit_intercept': ['True', 'False'],
        'normalize': ['True', 'False'],
        'n_jobs': [-1, 1]
    }
    tuned_model = GridSearchCV(model, model_params, cv=5, scoring='r2')
    tuned_model.fit(x_train, y_train)
    y_test_pred = tuned_model.predict(x_test)
    y_train_pred = tuned_model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Ethereum Prices','Predicted Training Prices','Predicted Prices'],'LR ETH Price Prediction')

def linear_regression_ltc(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = LinearRegression()
    #Tune model
    model_params = {
        'fit_intercept': ['True', 'False'],
        'normalize': ['True', 'False'],
        'n_jobs': [-1, 1]
    }
    tuned_model = GridSearchCV(model, model_params, cv=5, scoring='r2')
    tuned_model.fit(x_train, y_train)
    y_test_pred = tuned_model.predict(x_test)
    y_train_pred = tuned_model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Litecoin Prices','Predicted Training Prices','Predicted Prices'],'LR LTC Price Prediction')

def linear_regression_ada(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = LinearRegression()
    #Tune model
    model_params = {
        'fit_intercept': ['True', 'False'],
        'normalize': ['True', 'False'],
        'n_jobs': [-1, 1]
    }
    tuned_model = GridSearchCV(model, model_params, cv=5, scoring='r2')
    tuned_model.fit(x_train, y_train)
    y_test_pred = tuned_model.predict(x_test)
    y_train_pred = tuned_model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Cardano Prices','Predicted Training Prices','Predicted Prices'],'LR ADA Price Prediction')

def linear_regression_xrp(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = LinearRegression()
    #Tune model
    model_params = {
        'fit_intercept': ['True', 'False'],
        'normalize': ['True', 'False'],
        'n_jobs': [-1, 1]
    }
    tuned_model = GridSearchCV(model, model_params, cv=5, scoring='r2')
    tuned_model.fit(x_train, y_train)
    y_test_pred = tuned_model.predict(x_test)
    y_train_pred = tuned_model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Ripple Prices','Predicted Training Prices','Predicted Prices'],'LR XRP Price Prediction')

def linear_regression_xlm(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = LinearRegression()
    #Tune model
    model_params = {
        'fit_intercept': ['True', 'False'],
        'normalize': ['True', 'False'],
        'n_jobs': [-1, 1]
    }
    tuned_model = GridSearchCV(model, model_params, cv=5, scoring='r2')
    tuned_model.fit(x_train, y_train)
    y_test_pred = tuned_model.predict(x_test)
    y_train_pred = tuned_model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Stellar Lummens Prices','Predicted Training Prices','Predicted Prices'],'LR XLM Price Prediction')

def lstm_btc(dataset):
    #Split the data into training and testing set by date
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    dataset = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prediction_range = 10
    dataset.columns = names
    dataset['Prediction'] = dataset[['Close']].shift(-prediction_range)
    prices = np.array(dataset.drop(['Open','High','Low','Prediction','Adj Close','Volume'], axis=1).values)
    predictions = np.array(dataset.drop(['Open','High','Low','Close','Adj Close','Volume'], axis=1).values)
    timestep_size = 10
    #Scale the data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    predictions_scaled = scaler.fit_transform(predictions)
    scale = 1/scaler.scale_[0]
    #Split the data into x and y
    train_size = int(len(prices) * 0.8)
    test_size = int(len(prices_scaled) - train_size)
    prices_train_set, prices_test_set = prices_scaled[0:train_size,-prediction_range:], prices_scaled[train_size:len(prices_scaled)-prediction_range]
    predictions_train_set, predictions_test_set = predictions_scaled[0:train_size,-prediction_range:], predictions_scaled[train_size:len(predictions_scaled)-prediction_range]
    #Create datasets for lstm
    prices_x_train, prices_y_train = create_datasets(prices_train_set, timestep_size)
    prices_x_test, prices_y_test = create_datasets(prices_test_set, timestep_size)
    predictions_x_train, predictions_y_train = create_datasets(predictions_train_set, timestep_size)
    predictions_x_test, predictions_y_test = create_datasets(predictions_test_set, timestep_size)
    #Create model
    model = Sequential()
    model.add(LSTM(units=50, activation='linear', return_sequences=True))
    model.add(Dropout(.35))
    model.add(LSTM(units=100, activation='linear', return_sequences=False))
    model.add(Dropout(.35))
    model.add(Dense(units=1))
    #model.summary()
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mean_absolute_error'])
    model.fit(prices_x_train, prices_y_train, batch_size=64, epochs=20)
    #Predict the data
    y_train_pred = model.predict(predictions_x_train)
    y_test_pred = model.predict(predictions_x_test)
    print("R2 Testing Score: ",r2_score(prices_y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(prices_y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(prices_y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(prices_y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(prices_y_test, y_test_pred))
    #Unscale data
    y_train_pred = y_train_pred * scale
    y_test_pred = y_test_pred * scale
    #Plot data
    new_y_train_pred = [np.nan] * len(+prices)
    new_y_test_pred = [np.nan] * len(+prices)
    new_y_train_pred[timestep_size:train_size] = y_train_pred
    new_y_test_pred[train_size+timestep_size+1:] = y_test_pred
    plot_data(prices, new_y_train_pred, new_y_test_pred, ['Actual Bitcoin Prices','Predicted Training Prices','Predicted Testing Prices'], 'LSTM Bitcoin Price Prediction')

def lstm_eth(dataset):
    #Split the data into training and testing set by date
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    dataset = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prediction_range = 10
    dataset.columns = names
    dataset['Prediction'] = dataset[['Close']].shift(-prediction_range)
    prices = np.array(dataset.drop(['Open','High','Low','Prediction','Adj Close','Volume'], axis=1).values)
    predictions = np.array(dataset.drop(['Open','High','Low','Close','Adj Close','Volume'], axis=1).values)
    timestep_size = 10
    #Scale the data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    predictions_scaled = scaler.fit_transform(predictions)
    scale = 1/scaler.scale_[0]
    #Split the data into x and y
    train_size = int(len(prices) * 0.8)
    test_size = int(len(prices_scaled) - train_size)
    prices_train_set, prices_test_set = prices_scaled[0:train_size,-prediction_range:], prices_scaled[train_size:len(prices_scaled)-prediction_range]
    predictions_train_set, predictions_test_set = predictions_scaled[0:train_size,-prediction_range:], predictions_scaled[train_size:len(predictions_scaled)-prediction_range]
    #Create datasets for lstm
    prices_x_train, prices_y_train = create_datasets(prices_train_set, timestep_size)
    prices_x_test, prices_y_test = create_datasets(prices_test_set, timestep_size)
    predictions_x_train, predictions_y_train = create_datasets(predictions_train_set, timestep_size)
    predictions_x_test, predictions_y_test = create_datasets(predictions_test_set, timestep_size)
    #Create model
    model = Sequential()
    model.add(LSTM(units=50, activation='linear', return_sequences=True))
    model.add(Dropout(.35))
    model.add(LSTM(units=100, activation='linear', return_sequences=False))
    model.add(Dropout(.35))
    model.add(Dense(units=1))
    #model.summary()
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mean_absolute_error'])
    model.fit(prices_x_train, prices_y_train, batch_size=64, epochs=20)
    #Predict the data
    y_train_pred = model.predict(predictions_x_train)
    y_test_pred = model.predict(predictions_x_test)
    print("R2 Testing Score: ",r2_score(prices_y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(prices_y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(prices_y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(prices_y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(prices_y_test, y_test_pred))
    #Unscale data
    y_train_pred = y_train_pred * scale
    y_test_pred = y_test_pred * scale
    #Plot data
    new_y_train_pred = [np.nan] * len(+prices)
    new_y_test_pred = [np.nan] * len(+prices)
    new_y_train_pred[timestep_size:train_size] = y_train_pred
    new_y_test_pred[train_size+timestep_size+1:] = y_test_pred
    plot_data(prices, new_y_train_pred, new_y_test_pred, ['Actual Ethereum Prices','Predicted Training Prices','Predicted Testing Prices'], 'LSTM ETH Price Prediction')

def lstm_ltc(dataset):
    #Split the data into training and testing set by date
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    dataset = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prediction_range = 10
    dataset.columns = names
    dataset['Prediction'] = dataset[['Close']].shift(-prediction_range)
    prices = np.array(dataset.drop(['Open','High','Low','Prediction','Adj Close','Volume'], axis=1).values)
    predictions = np.array(dataset.drop(['Open','High','Low','Close','Adj Close','Volume'], axis=1).values)
    timestep_size = 10
    #Scale the data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    predictions_scaled = scaler.fit_transform(predictions)
    scale = 1/scaler.scale_[0]
    #Split the data into x and y
    train_size = int(len(prices) * 0.8)
    test_size = int(len(prices_scaled) - train_size)
    prices_train_set, prices_test_set = prices_scaled[0:train_size,-prediction_range:], prices_scaled[train_size:len(prices_scaled)-prediction_range]
    predictions_train_set, predictions_test_set = predictions_scaled[0:train_size,-prediction_range:], predictions_scaled[train_size:len(predictions_scaled)-prediction_range]
    #Create datasets for lstm
    prices_x_train, prices_y_train = create_datasets(prices_train_set, timestep_size)
    prices_x_test, prices_y_test = create_datasets(prices_test_set, timestep_size)
    predictions_x_train, predictions_y_train = create_datasets(predictions_train_set, timestep_size)
    predictions_x_test, predictions_y_test = create_datasets(predictions_test_set, timestep_size)
    #Create model
    model = Sequential()
    model.add(LSTM(units=50, activation='linear', return_sequences=True))
    model.add(Dropout(.35))
    model.add(LSTM(units=100, activation='linear', return_sequences=False))
    model.add(Dropout(.35))
    model.add(Dense(units=1))
    #model.summary()
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mean_absolute_error'])
    model.fit(prices_x_train, prices_y_train, batch_size=64, epochs=20)
    #Predict the data
    y_train_pred = model.predict(predictions_x_train)
    y_test_pred = model.predict(predictions_x_test)
    print("R2 Testing Score: ",r2_score(prices_y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(prices_y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(prices_y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(prices_y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(prices_y_test, y_test_pred))
    #Unscale data
    y_train_pred = y_train_pred * scale
    y_test_pred = y_test_pred * scale
    #Plot data
    new_y_train_pred = [np.nan] * len(+prices)
    new_y_test_pred = [np.nan] * len(+prices)
    new_y_train_pred[timestep_size:train_size] = y_train_pred
    new_y_test_pred[train_size+timestep_size+1:] = y_test_pred
    plot_data(prices, new_y_train_pred, new_y_test_pred, ['Actual Litecoin Prices','Predicted Training Prices','Predicted Testing Prices'], 'LSTM LTC Price Prediction')

def lstm_ada(dataset):
    #Split the data into training and testing set by date
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    dataset = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prediction_range = 10
    dataset.columns = names
    dataset['Prediction'] = dataset[['Close']].shift(-prediction_range)
    prices = np.array(dataset.drop(['Open','High','Low','Prediction','Adj Close','Volume'], axis=1).values)
    predictions = np.array(dataset.drop(['Open','High','Low','Close','Adj Close','Volume'], axis=1).values)
    timestep_size = 10
    #Scale the data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    predictions_scaled = scaler.fit_transform(predictions)
    scale = 1/scaler.scale_[0]
    #Split the data into x and y
    train_size = int(len(prices) * 0.8)
    test_size = int(len(prices_scaled) - train_size)
    prices_train_set, prices_test_set = prices_scaled[0:train_size,-prediction_range:], prices_scaled[train_size:len(prices_scaled)-prediction_range]
    predictions_train_set, predictions_test_set = predictions_scaled[0:train_size,-prediction_range:], predictions_scaled[train_size:len(predictions_scaled)-prediction_range]
    #Create datasets for lstm
    prices_x_train, prices_y_train = create_datasets(prices_train_set, timestep_size)
    prices_x_test, prices_y_test = create_datasets(prices_test_set, timestep_size)
    predictions_x_train, predictions_y_train = create_datasets(predictions_train_set, timestep_size)
    predictions_x_test, predictions_y_test = create_datasets(predictions_test_set, timestep_size)
    #Create model
    model = Sequential()
    model.add(LSTM(units=50, activation='linear', return_sequences=True))
    model.add(Dropout(.35))
    model.add(LSTM(units=100, activation='linear', return_sequences=False))
    model.add(Dropout(.35))
    model.add(Dense(units=1))
    #model.summary()
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mean_absolute_error'])
    model.fit(prices_x_train, prices_y_train, batch_size=64, epochs=20)
    #Predict the data
    y_train_pred = model.predict(predictions_x_train)
    y_test_pred = model.predict(predictions_x_test)
    print("R2 Testing Score: ",r2_score(prices_y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(prices_y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(prices_y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(prices_y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(prices_y_test, y_test_pred))
    #Unscale data
    y_train_pred = y_train_pred * scale
    y_test_pred = y_test_pred * scale
    #Plot data
    new_y_train_pred = [np.nan] * len(+prices)
    new_y_test_pred = [np.nan] * len(+prices)
    new_y_train_pred[timestep_size:train_size] = y_train_pred
    new_y_test_pred[train_size+timestep_size+1:] = y_test_pred
    plot_data(prices, new_y_train_pred, new_y_test_pred, ['Actual Cardano Prices','Predicted Training Prices','Predicted Testing Prices'], 'LSTM ADA Price Prediction')

def lstm_xrp(dataset):
    #Split the data into training and testing set by date
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    dataset = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prediction_range = 10
    dataset.columns = names
    dataset['Prediction'] = dataset[['Close']].shift(-prediction_range)
    prices = np.array(dataset.drop(['Open','High','Low','Prediction','Adj Close','Volume'], axis=1).values)
    predictions = np.array(dataset.drop(['Open','High','Low','Close','Adj Close','Volume'], axis=1).values)
    timestep_size = 10
    #Scale the data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    predictions_scaled = scaler.fit_transform(predictions)
    scale = 1/scaler.scale_[0]
    #Split the data into x and y
    train_size = int(len(prices) * 0.8)
    test_size = int(len(prices_scaled) - train_size)
    prices_train_set, prices_test_set = prices_scaled[0:train_size,-prediction_range:], prices_scaled[train_size:len(prices_scaled)-prediction_range]
    predictions_train_set, predictions_test_set = predictions_scaled[0:train_size,-prediction_range:], predictions_scaled[train_size:len(predictions_scaled)-prediction_range]
    #Create datasets for lstm
    prices_x_train, prices_y_train = create_datasets(prices_train_set, timestep_size)
    prices_x_test, prices_y_test = create_datasets(prices_test_set, timestep_size)
    predictions_x_train, predictions_y_train = create_datasets(predictions_train_set, timestep_size)
    predictions_x_test, predictions_y_test = create_datasets(predictions_test_set, timestep_size)
    #Create model
    model = Sequential()
    model.add(LSTM(units=50, activation='linear', return_sequences=True))
    model.add(Dropout(.35))
    model.add(LSTM(units=100, activation='linear', return_sequences=False))
    model.add(Dropout(.35))
    model.add(Dense(units=1))
    #model.summary()
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mean_absolute_error'])
    model.fit(prices_x_train, prices_y_train, batch_size=64, epochs=20)
    #Predict the data
    y_train_pred = model.predict(predictions_x_train)
    y_test_pred = model.predict(predictions_x_test)
    print("R2 Testing Score: ",r2_score(prices_y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(prices_y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(prices_y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(prices_y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(prices_y_test, y_test_pred))
    #Unscale data
    y_train_pred = y_train_pred * scale
    y_test_pred = y_test_pred * scale
    #Plot data
    new_y_train_pred = [np.nan] * len(+prices)
    new_y_test_pred = [np.nan] * len(+prices)
    new_y_train_pred[timestep_size:train_size] = y_train_pred
    new_y_test_pred[train_size+timestep_size+1:] = y_test_pred
    plot_data(prices, new_y_train_pred, new_y_test_pred, ['Actual Ripple Prices','Predicted Training Prices','Predicted Testing Prices'], 'LSTM XRP Price Prediction')

def lstm_xlm(dataset):
    #Split the data into training and testing set by date
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    dataset = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prediction_range = 10
    dataset.columns = names
    dataset['Prediction'] = dataset[['Close']].shift(-prediction_range)
    prices = np.array(dataset.drop(['Open','High','Low','Prediction','Adj Close','Volume'], axis=1).values)
    predictions = np.array(dataset.drop(['Open','High','Low','Close','Adj Close','Volume'], axis=1).values)
    timestep_size = 10
    #Scale the data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    predictions_scaled = scaler.fit_transform(predictions)
    scale = 1/scaler.scale_[0]
    #Split the data into x and y
    train_size = int(len(prices) * 0.8)
    test_size = int(len(prices_scaled) - train_size)
    prices_train_set, prices_test_set = prices_scaled[0:train_size,-prediction_range:], prices_scaled[train_size:len(prices_scaled)-prediction_range]
    predictions_train_set, predictions_test_set = predictions_scaled[0:train_size,-prediction_range:], predictions_scaled[train_size:len(predictions_scaled)-prediction_range]
    #Create datasets for lstm
    prices_x_train, prices_y_train = create_datasets(prices_train_set, timestep_size)
    prices_x_test, prices_y_test = create_datasets(prices_test_set, timestep_size)
    predictions_x_train, predictions_y_train = create_datasets(predictions_train_set, timestep_size)
    predictions_x_test, predictions_y_test = create_datasets(predictions_test_set, timestep_size)
    #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    #Create model
    model = Sequential()
    model.add(LSTM(units=50, activation='linear', return_sequences=True))
    model.add(Dropout(.35))
    model.add(LSTM(units=100, activation='linear', return_sequences=False))
    model.add(Dropout(.35))
    model.add(Dense(units=1))
    #model.summary()
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mean_absolute_error'])
    model.fit(prices_x_train, prices_y_train, batch_size=64, epochs=20)
    #Predict the data
    y_train_pred = model.predict(predictions_x_train)
    y_test_pred = model.predict(predictions_x_test)
    print("R2 Testing Score: ",r2_score(prices_y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(prices_y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(prices_y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(prices_y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(prices_y_test, y_test_pred))
    #Unscale data
    y_train_pred = y_train_pred * scale
    y_test_pred = y_test_pred * scale
    #Plot data
    new_y_train_pred = [np.nan] * len(+prices)
    new_y_test_pred = [np.nan] * len(+prices)
    new_y_train_pred[timestep_size:train_size] = y_train_pred
    new_y_test_pred[train_size+timestep_size+1:] = y_test_pred
    plot_data(prices, new_y_train_pred, new_y_test_pred, ['Actual Stellar Lummens Prices','Predicted Training Prices','Predicted Testing Prices'], 'LSTM XLM Price Prediction')

def svr_btc(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = SVR(kernel='poly', degree=2, coef0=2, C=0.25, epsilon=0.8)
    model.fit(x_train,y_train)
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Bitcoin Prices','Predicted Training Prices','Predicted Testing Prices'], 'SVR Bitcoin Price Prediction')

def svr_eth(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    tuned_model = SVR(kernel='poly', degree=2, coef0=2, C=0.25, epsilon=0.8)
    tuned_model.fit(x_train, y_train)
    y_test_pred = tuned_model.predict(x_test)
    y_train_pred = tuned_model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Ethereum Prices','Predicted Training Prices','Predicted Testing Prices'], 'SVR ETH Price Prediction')

def svr_ltc(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    tuned_model = SVR(kernel='poly', degree=2, coef0=2, C=0.25, epsilon=0.8)
    tuned_model.fit(x_train, y_train)
    y_test_pred = tuned_model.predict(x_test)
    y_train_pred = tuned_model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Litecoin Prices','Predicted Training Prices','Predicted Testing Prices'], 'SVR LTC Price Prediction')

def svr_ada(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = SVR(kernel='linear',gamma='auto',epsilon=0.0,coef0=0.0,C=0.5)
    model.fit(x_train,y_train)
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Cardano Prices','Predicted Training Prices','Predicted Testing Prices'], 'SVR ADA Price Prediction')

def svr_xrp(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    tuned_model = SVR(kernel='linear',gamma='auto',epsilon=0.0,coef0=0.0,C=0.5)
    tuned_model.fit(x_train, y_train)
    y_test_pred = tuned_model.predict(x_test)
    y_train_pred = tuned_model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Ripple Prices','Predicted Training Prices','Predicted Testing Prices'], 'SVR XRP Price Prediction')

def svr_xlm(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    tuned_model = SVR(kernel='linear',gamma='auto',epsilon=0.0,coef0=0.0,C=0.5)
    tuned_model.fit(x_train, y_train)
    y_test_pred = tuned_model.predict(x_test)
    y_train_pred = tuned_model.predict(x_train)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Stellar Lummens Prices','Predicted Training Prices','Predicted Testing Prices'], 'SVR XLM Price Prediction')

#neural networks
def neural_networks_btc(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = MLPRegressor(hidden_layer_sizes=(1000,), activation='identity', solver='adam', alpha=0.01, batch_size=50, learning_rate='adaptive', max_iter=1000)
    #print(tuned_model.cv_results_)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Bitcoin Prices','Predicted Training Prices','Predicted Testing Prices'], 'MLPR(NN) Bitcoin Price Prediction')

def neural_networks_eth(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = MLPRegressor(hidden_layer_sizes=(1000,), activation='identity', solver='adam', alpha=0.01, batch_size=50, learning_rate='adaptive', max_iter=1000)
    #print(tuned_model.cv_results_)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Ethereum Prices','Predicted Training Prices','Predicted Testing Prices'], 'MLPR(NN) ETH Price Prediction')

def neural_networks_ltc(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = MLPRegressor(hidden_layer_sizes=(1000,), activation='identity', solver='adam', alpha=0.01, batch_size=50, learning_rate='adaptive', max_iter=1000)
    #print(tuned_model.cv_results_)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Litecoin Prices','Predicted Training Prices','Predicted Testing Prices'], 'MLPR(NN) LTC Price Prediction')

def neural_networks_ada(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = MLPRegressor(hidden_layer_sizes=(1000,), activation='identity', solver='adam', alpha=0.01, batch_size=50, learning_rate='adaptive', max_iter=1000)
    #print(tuned_model.cv_results_)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Cardano Prices','Predicted Training Prices','Predicted Testing Prices'], 'MLPR(NN) ADA Price Prediction')

def neural_networks_xrp(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    tuned_model = MLPRegressor(hidden_layer_sizes=(500,), activation='identity', solver='lbfgs', alpha=0.02, batch_size=60, learning_rate='constant', max_iter=1000)
    tuned_model.fit(x_train, y_train)
    y_train_pred = tuned_model.predict(x_train)
    y_test_pred = tuned_model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Ripple Prices','Predicted Training Prices','Predicted Testing Prices'], 'MLPR(NN) XRP Price Prediction')

def neural_networks_xlm(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    tuned_model = MLPRegressor(hidden_layer_sizes=(500,), activation='identity', solver='lbfgs', alpha=0.02, batch_size=60, learning_rate='constant', max_iter=1000)
    tuned_model.fit(x_train, y_train)
    y_train_pred = tuned_model.predict(x_train)
    y_test_pred = tuned_model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Stellar Lummens Prices','Predicted Training Prices','Predicted Testing Prices'], 'MLPR(NN) XLM Price Prediction')

#random forests
def extra_trees_btc(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model#model = RandomForestRegressor(criterion='mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0, max_features='auto', bootstrap=False, warm_start=True)
    model = ExtraTreesRegressor()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Bitcoin Prices','Predicted Training Prices','Predicted Testing Prices'], 'ET Bitcoin Price Prediction')
    
def extra_trees_eth(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = ExtraTreesRegressor()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Ethereum Prices','Predicted Training Prices','Predicted Testing Prices'], 'ET ETH Price Prediction')

def extra_trees_ltc(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = ExtraTreesRegressor()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Litecoin Prices','Predicted Training Prices','Predicted Testing Prices'], 'ET LTC Price Prediction')

def extra_trees_ada(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = ExtraTreesRegressor()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Cardano Prices','Predicted Training Prices','Predicted Testing Prices'], 'ET ADA Price Prediction')

def extra_trees_xrp(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = ExtraTreesRegressor()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Ripple Prices','Predicted Training Prices','Predicted Testing Prices'], 'ET XRP Price Prediction')

def extra_trees_xlm(dataset):
    x, y, x_train, x_test, y_train, y_test = split_data(dataset)
    #Create model
    model = ExtraTreesRegressor()
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("R2 Testing Score: ",r2_score(y_test, y_test_pred))
    print("Explained Variance Testing Score: ",explained_variance_score(y_test, y_test_pred))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_test, y_test_pred))
    print("Mean Squared Testing Error: ", mean_squared_error(y_test, y_test_pred))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_test, y_test_pred))
    new_y_train_pred, new_y_test_pred = prepare_plots(y, y_train, y_train_pred, y_test_pred)
    plot_data(y, new_y_train_pred, new_y_test_pred, ['Actual Stellar Lummens Prices','Predicted Training Prices','Predicted Testing Prices'], 'ET XLM Price Prediction')
                    
def incremental_model_daily_btc(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 10
    test_size = int(len(prices)*0.2)
    prices['Prediction'] = prices[['Close']].shift(-prediction_range)
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close','Prediction'], axis=1))
    x_true = x[:len(prices) - prediction_range]
    y = np.array(prices['Prediction'])
    y_true = y[: -prediction_range]
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    model = HoeffdingAdaptiveTreeRegressor()
    n_samples = 0
    correct_pred = 0
    pred_prices = []
    while n_samples < x_scaled.shape[0]-30:
        x, y = [x_true[n_samples]], [y_true[n_samples]]
        model.partial_fit(x,y)
        pred = model.predict(x)
        if y == pred:
            correct_pred += 1
        n_samples += 1
        pred_prices.append(pred)
    print("R2 Testing Score: ",r2_score(y_true, pred_prices))
    print("Explained Variance Testing Score: ",explained_variance_score(y_true, pred_prices))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_true, pred_prices))
    print("Mean Squared Testing Error: ", mean_squared_error(y_true, pred_prices))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_true, pred_prices))
    plt.figure(figsize=(14,6))
    plt.plot(y_true, 'b', label='Actual Prices')
    plt.plot(pred_prices, 'r', label='Predicted Prices')
    plt.legend(loc='upper left')
    plt.show()

def incremental_model_daily_eth(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 10
    test_size = int(len(prices)*0.2)
    prices['Prediction'] = prices[['Close']].shift(-prediction_range)
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close','Prediction'], axis=1))
    x_true = x[:len(prices) - prediction_range]
    y = np.array(prices['Prediction'])
    y_true = y[: -prediction_range]
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    model = HoeffdingAdaptiveTreeRegressor()
    n_samples = 0
    correct_pred = 0
    pred_prices = []
    while n_samples < x_scaled.shape[0]-30:
        x, y = [x_true[n_samples]], [y_true[n_samples]]
        model.partial_fit(x,y)
        pred = model.predict(x)
        if y == pred:
            correct_pred += 1
        n_samples += 1
        pred_prices.append(pred)
    print("R2 Testing Score: ",r2_score(y_true, pred_prices))
    print("Explained Variance Testing Score: ",explained_variance_score(y_true, pred_prices))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_true, pred_prices))
    print("Mean Squared Testing Error: ", mean_squared_error(y_true, pred_prices))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_true, pred_prices))
    plt.figure(figsize=(14,6))
    plt.plot(y_true, 'b', label='Actual Prices')
    plt.plot(pred_prices, 'r', label='Predicted Prices')
    plt.legend(loc='upper left')
    plt.show()

def incremental_model_daily_ltc(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 10
    test_size = int(len(prices)*0.2)
    prices['Prediction'] = prices[['Close']].shift(-prediction_range)
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close','Prediction'], axis=1))
    x_true = x[:len(prices) - prediction_range]
    y = np.array(prices['Prediction'])
    y_true = y[: -prediction_range]
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    model = HoeffdingAdaptiveTreeRegressor()
    n_samples = 0
    correct_pred = 0
    pred_prices = []
    while n_samples < x_scaled.shape[0]-30:
        x, y = [x_true[n_samples]], [y_true[n_samples]]
        model.partial_fit(x,y)
        pred = model.predict(x)
        if y == pred:
            correct_pred += 1
        n_samples += 1
        pred_prices.append(pred)
    print("R2 Testing Score: ",r2_score(y_true, pred_prices))
    print("Explained Variance Testing Score: ",explained_variance_score(y_true, pred_prices))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_true, pred_prices))
    print("Mean Squared Testing Error: ", mean_squared_error(y_true, pred_prices))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_true, pred_prices))
    plt.figure(figsize=(14,6))
    plt.plot(y_true, 'b', label='Actual Prices')
    plt.plot(pred_prices, 'r', label='Predicted Prices')
    plt.legend(loc='upper left')
    plt.show()

def incremental_model_daily_ada(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 10
    test_size = int(len(prices)*0.2)
    prices['Prediction'] = prices[['Close']].shift(-prediction_range)
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close','Prediction'], axis=1))
    x_true = x[:len(prices) - prediction_range]
    y = np.array(prices['Prediction'])
    y_true = y[: -prediction_range]
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    model = HoeffdingAdaptiveTreeRegressor()
    n_samples = 0
    correct_pred = 0
    pred_prices = []
    while n_samples < x_scaled.shape[0]-30:
        x, y = [x_true[n_samples]], [y_true[n_samples]]
        model.partial_fit(x,y)
        pred = model.predict(x)
        if y == pred:
            correct_pred += 1
        n_samples += 1
        pred_prices.append(pred)
    print("R2 Testing Score: ",r2_score(y_true, pred_prices))
    print("Explained Variance Testing Score: ",explained_variance_score(y_true, pred_prices))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_true, pred_prices))
    print("Mean Squared Testing Error: ", mean_squared_error(y_true, pred_prices))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_true, pred_prices))
    plt.figure(figsize=(14,6))
    plt.plot(y_true, 'b', label='Actual Prices')
    plt.plot(pred_prices, 'r', label='Predicted Prices')
    plt.legend(loc='upper left')
    plt.show()

def incremental_model_daily_xrp(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 10
    test_size = int(len(prices)*0.2)
    prices['Prediction'] = prices[['Close']].shift(-prediction_range)
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close','Prediction'], axis=1))
    x_true = x[:len(prices) - prediction_range]
    y = np.array(prices['Prediction'])
    y_true = y[: -prediction_range]
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    model = HoeffdingAdaptiveTreeRegressor()
    n_samples = 0
    correct_pred = 0
    pred_prices = []
    while n_samples < x_scaled.shape[0]-30:
        x, y = [x_true[n_samples]], [y_true[n_samples]]
        model.partial_fit(x,y)
        pred = model.predict(x)
        if y == pred:
            correct_pred += 1
        n_samples += 1
        pred_prices.append(pred)
    print("R2 Testing Score: ",r2_score(y_true, pred_prices))
    print("Explained Variance Testing Score: ",explained_variance_score(y_true, pred_prices))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_true, pred_prices))
    print("Mean Squared Testing Error: ", mean_squared_error(y_true, pred_prices))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_true, pred_prices))
    plt.figure(figsize=(14,6))
    plt.plot(y_true, 'b', label='Actual Prices')
    plt.plot(pred_prices, 'r', label='Predicted Prices')
    plt.legend(loc='upper left')
    plt.show()

def incremental_model_daily_xlm(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 10
    test_size = int(len(prices)*0.2)
    prices['Prediction'] = prices[['Close']].shift(-prediction_range)
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close','Prediction'], axis=1))
    x_true = x[:len(prices) - prediction_range]
    y = np.array(prices['Prediction'])
    y_true = y[: -prediction_range]
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    model = HoeffdingAdaptiveTreeRegressor()
    n_samples = 0
    correct_pred = 0
    pred_prices = []
    while n_samples < x_scaled.shape[0]-30:
        x, y = [x_true[n_samples]], [y_true[n_samples]]
        model.partial_fit(x,y)
        pred = model.predict(x)
        if y == pred:
            correct_pred += 1
        n_samples += 1
        pred_prices.append(pred)
    print("R2 Testing Score: ",r2_score(y_true, pred_prices))
    print("Explained Variance Testing Score: ",explained_variance_score(y_true, pred_prices))
    print("Mean Absolute Testing Error: ", mean_absolute_error(y_true, pred_prices))
    print("Mean Squared Testing Error: ", mean_squared_error(y_true, pred_prices))
    print("Mean Absolute Percentage Testing Error: ", mean_absolute_percentage_error(y_true, pred_prices))
    plt.figure(figsize=(14,6))
    plt.plot(y_true, 'b', label='Actual Prices')
    plt.plot(pred_prices, 'r', label='Predicted Prices')
    plt.legend(loc='upper left')
    plt.show()

"""def incremental_model_minute_btc():
    model = HoeffdingAdaptiveTreeRegressor()
    pred_prices = []
    actual_prices = []
    while True:
        data = collect_price_data_btc()
        #x_scaled = scaler.fit_transform(x)
        #print(scaler.scale_)
        x, y = data.drop(['price'], axis=1).values, data['price'].values
        model.partial_fit(x, y)
        pred = model.predict(x)
        actual_prices.append(y[0])
        pred_prices.append(pred[0])
        print("Actual Price: {0}, Pred Price: {1}".format(y, pred))
        print(actual_prices, pred_prices)
        print("R2 Score: ",r2_score(actual_prices, pred_prices))
        #print(r2_score(actual_prices, pred_prices))"""