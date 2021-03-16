#Imports
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math
from matplotlib.legend_handler import HandlerLine2D
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split,  GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import max_error, r2_score, mean_squared_error
from sklearn.utils import shuffle
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

#linear regression
def linear_regression_btc(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 30
    test_size = 563
    prices['Prediction'] = prices[['Close']].shift(-prediction_range)
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close','Prediction'], axis=1))
    x = x[:len(prices) - prediction_range]
    y = np.array(prices['Prediction'])
    y = y[: -prediction_range]
    #Scale the data
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    #Split data into training and testing sets, -352 for 1st dataset
    x_train, x_test = x_scaled[:len(prices) - prediction_range - test_size], x_scaled[len(prices) - prediction_range - test_size:]
    y_train, y_test = y[:len(prices) - prediction_range - test_size], y[len(prices) - prediction_range - test_size:]
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    #Create model
    model = LinearRegression()
    #Tune model
    model_params = {
        'fit_intercept': ['True', 'False'],
        'normalize': ['True', 'False'],
        'n_jobs': [-1, 1]
    }
    tuned_model = GridSearchCV(model, model_params, cv=5, scoring='r2')
    #Score w/tuning: 0.9986911037149039
    #Score w/o tuning: 0.9986911037119561
    tuned_model.fit(x_train, y_train)
    print("Training Accuracy: ",tuned_model.score(x_train, y_train))
    print("Model Accuracy: ",tuned_model.score(x_test, y_test))
    y_pred = tuned_model.predict(x_test)
    new_y_pred = np.empty_like(y)
    new_y_pred[:] = np.nan
    new_y_pred[len(y_train):] = y_pred
    plt.figure(figsize=(14,6))
    plt.plot(y)
    plt.plot(new_y_pred, color='r')
    plt.show()

def lstm_btc(dataset):
    #Split the data into training and testing set by date
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    timestep_size = 10
    #Scale the data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    scale = 1/scaler.scale_[0]
    #Split the data into x and y
    train_size = 1799
    test_size = int(len(prices_scaled) - train_size)
    train_set, test_set = prices_scaled[0:train_size,:], prices_scaled[train_size:len(prices_scaled)]
    #Create datasets for lstm
    x_train, y_train = create_datasets(train_set, timestep_size)
    x_test, y_test = create_datasets(test_set, timestep_size)
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
    model_performance = model.fit(x_train, y_train, batch_size=64, epochs=10)
    """plt.figure(figsize=(10,5))
    plt.plot(model_performance.history['loss'])
    plt.plot(model_performance.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()"""
    #Predict the data
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print(y_test.shape, y_test_pred.shape)
    #Unscale data
    y_train_pred = y_train_pred * scale
    y_test_pred = y_test_pred * scale
    #Plot data
    new_y_train_pred = [np.nan] * len(+prices)
    new_y_test_pred = [np.nan] * len(+prices)
    new_y_train_pred[timestep_size:train_size] = y_train_pred
    new_y_test_pred[train_size+timestep_size+1:] = y_test_pred
    plt.figure(figsize=(14,6))
    plt.plot(prices['Close'], 'g', label='Actual Bitcoin Prices')
    predicted_train_prices = plt.plot(new_y_train_pred, 'r', label='Predicted Training Prices')
    predicted_test_prices = plt.plot(new_y_test_pred, 'b', label='Predicted Testing Prices')
    plt.xlabel('Time in days')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.title('Price prediction')
    plt.show()

def svr_btc(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 30
    test_size = 563
    prices['Prediction'] = prices[['Close']].shift(-prediction_range)
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close','Prediction'], axis=1))
    x = x[:len(prices) - prediction_range]
    y = np.array(prices['Prediction'])
    y = y[: -prediction_range]
    #Scale the data
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    #Split data into training and testing sets, -352 for 1st dataset
    x_train, x_test = x_scaled[:len(prices) - prediction_range - test_size], x_scaled[len(prices) - prediction_range - test_size:]
    y_train, y_test = y[:len(prices) - prediction_range - test_size], y[len(prices) - prediction_range - test_size:]
    #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    #Create model
    model = SVR(kernel='poly', degree=2, coef0=2, C=0.25, epsilon=0.8)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("Training Accuracy: ",model.score(x_train, y_train))
    print("Model Accuracy: ",model.score(x_test, y_test))
    new_y_pred = np.empty_like(y)
    new_y_pred[:] = np.nan
    new_y_pred[len(y_train):] = y_pred
    plt.figure(figsize=(14,6))
    plt.plot(y)
    plt.plot(new_y_pred, color='r')
    plt.show()
    
#neural networks
def neural_networks_btc(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 30
    test_size = 563
    prices['Prediction'] = prices[['Close']].shift(-prediction_range)
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close','Prediction'], axis=1))
    x = x[:len(prices) - prediction_range]
    y = np.array(prices['Prediction'])
    y = y[: -prediction_range]
    #Scale data
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    #Split data into training and testing sets, -352 for 1st dataset
    x_train, x_test = x_scaled[:len(prices) - prediction_range - test_size], x_scaled[len(prices) - prediction_range - test_size:]
    y_train, y_test = y[:len(prices) - prediction_range - test_size], y[len(prices) - prediction_range - test_size:]
    #Create model
    model = MLPRegressor(hidden_layer_sizes=(1000,), activation='identity', solver='adam', alpha=0.01, batch_size=50, learning_rate='adaptive', max_iter=1000)
    #print(tuned_model.cv_results_)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("Training Accuracy: ",model.score(x_train, y_train))
    print("Model Accuracy: ",model.score(x_test, y_test))
    new_y_train_pred = np.empty_like(y)
    new_y_train_pred[:] = np.nan
    new_y_train_pred[:len(y_train)] = y_train_pred
    new_y_test_pred = np.empty_like(y)
    new_y_test_pred[:] = np.nan
    new_y_test_pred[len(y_train):] = y_test_pred
    plt.figure(figsize=(14,6))
    plt.plot(y, color='b', label='Actual BTC Prices')
    plt.plot(new_y_train_pred, color='g', label='Predicted Training Set')
    plt.plot(new_y_test_pred, color='r', label='Predicted Testing Prices')
    plt.legend(loc='upper left')
    plt.show()

#random forests
def random_forest_btc(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 30
    test_size = 563
    prices['Prediction'] = prices[['Close']].shift(-prediction_range)
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close','Prediction'], axis=1))
    x = x[:len(prices) - prediction_range]
    y = np.array(prices['Prediction'])
    y = y[: -prediction_range]
    #Scale data
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    #Split data into training and testing sets, -352 for 1st dataset
    x_train, x_test = x_scaled[:len(prices) - prediction_range - test_size], x_scaled[len(prices) - prediction_range - test_size:]
    y_train, y_test = y[:len(prices) - prediction_range - test_size], y[len(prices) - prediction_range - test_size:]
    #Create model
    model = RandomForestRegressor(n_estimators=400, criterion='mae', max_depth=400)
    #print(tuned_model.cv_results_)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("Training Accuracy: ",model.score(x_train, y_train))
    print("Model Accuracy: ",model.score(x_test, y_test))
    new_y_train_pred = np.empty_like(y)
    new_y_train_pred[:] = np.nan
    new_y_train_pred[:len(y_train)] = y_train_pred
    new_y_test_pred = np.empty_like(y)
    new_y_test_pred[:] = np.nan
    new_y_test_pred[len(y_train):] = y_test_pred
    plt.figure(figsize=(14,6))
    plt.plot(y, color='b', label='Actual BTC Prices')
    plt.plot(new_y_train_pred, color='g', label='Predicted Training Set')
    plt.plot(new_y_test_pred, color='r', label='Predicted Testing Prices')
    plt.legend(loc='upper left')
    plt.show()
    
#TODO: Try and create a model that collects data on daily basis and make it recurring.
def progressive_lr_prediction(x,y):
    pass