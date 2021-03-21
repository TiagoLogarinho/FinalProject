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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split,  GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import max_error, r2_score, mean_squared_error
from sklearn.utils import shuffle
from skmultiflow.neural_networks import PerceptronMask
from skmultiflow.trees import HoeffdingTreeRegressor, HoeffdingAdaptiveTreeRegressor, StackedSingleTargetHoeffdingTreeRegressor
from skmultiflow.data import FileStream
from skmultiflow.data import SEAGenerator
###########################################

def collect_price_data_btc():
    PATH = './chromedriver.exe'
    options = Options()
    options.add_argument('--log-level=3')
    options.add_argument('--headless')
    driver = webdriver.Chrome(PATH, options=options)
    url = 'https://www.coindesk.com/price/bitcoin'

    driver.get(url)
    driver.find_element_by_class_name('dropdown-header-title').click()
    driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[2]/div/div/div/ul/li[2]').click()

    while True:
        now = datetime.now().strftime('%S')
        if now == '00':
            open = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[4]/div[2]/div').text[1:]
            high = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[2]/div[2]/div').text[1:]
            low = driver.find_element_by_xpath('//*[@id="__next"]/div[2]/main/section/div/div[1]/div/section/div/div[3]/div/div[2]/div[1]/div[2]/div').text[1:]
            volume = driver.find_element_by_xpath('//*[@id="export-chart-element"]/div/section/div[1]/div[4]/div[2]/div').text[1:-1]
            price = driver.find_element_by_class_name('price-large').text[1:]
            open = open.replace(',','')
            high = high.replace(',','')
            low = low.replace(',','')
            volume = float(volume) * 1000000000
            price = price.replace(',','')
            x = np.array([float(open), float(high), float(low), int(volume)])
            y = np.array([float(price)])
            return x, y
    

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
    y_test_pred = tuned_model.predict(x_test)
    y_train_pred = tuned_model.predict(x_train)
    print("R2 Score: ", r2_score(y_test, y_test_pred))
    new_y_test_pred = np.empty_like(y)
    new_y_train_pred = np.empty_like(y)
    new_y_test_pred[:] = np.nan
    new_y_train_pred[:] = np.nan
    new_y_test_pred[len(y_train):] = y_test_pred
    new_y_train_pred[:len(y_train)] = y_train_pred
    plt.figure(figsize=(14,6))
    plt.plot(y, color='b', label='Actual BTC Prices')
    plt.plot(new_y_test_pred, color='r', label='Predicted Prices')
    plt.plot(new_y_train_pred, color='g', label='Predicted Training Prices')
    plt.legend(loc='upper left')
    plt.show()

def lstm_btc(dataset):
    #Split the data into training and testing set by date
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    dataset = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prediction_range = 30
    dataset.columns = names
    dataset['Prediction'] = dataset[['Close']].shift(-prediction_range)
    prices = np.array(dataset.drop(['Open','High','Low','Prediction','Adj Close','Volume'], axis=1).values)
    print(prices)
    predictions = np.array(dataset.drop(['Open','High','Low','Close','Adj Close','Volume'], axis=1).values)
    #prices.columns = names
    timestep_size = 10
    #Scale the data
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    predictions_scaled = scaler.fit_transform(predictions)
    scale = 1/scaler.scale_[0]
    #Split the data into x and y
    train_size = 1799
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
    model_performance = model.fit(prices_x_train, prices_y_train, batch_size=64, epochs=10)
    """plt.figure(figsize=(10,5))
    plt.plot(model_performance.history['loss'])
    plt.plot(model_performance.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()"""
    #Predict the data
    y_train_pred = model.predict(predictions_x_train)
    y_test_pred = model.predict(predictions_x_test)
    print(predictions_y_test.shape, y_test_pred.shape)
    print("R2 Score: ",r2_score(predictions_y_test, y_test_pred))
    #Unscale data
    y_train_pred = y_train_pred * scale
    y_test_pred = y_test_pred * scale
    #Plot data
    new_y_train_pred = [np.nan] * len(+prices)
    new_y_test_pred = [np.nan] * len(+prices)
    new_y_train_pred[timestep_size:train_size] = y_train_pred
    new_y_test_pred[train_size+timestep_size+1:] = y_test_pred
    plt.figure(figsize=(14,6))
    plt.plot(prices, 'b', label='Actual Bitcoin Prices')
    plt.plot(new_y_train_pred, 'g', label='Predicted Training Prices')
    plt.plot(new_y_test_pred, 'r', label='Predicted Testing Prices')
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
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    print("Training Accuracy: ",model.score(x_train, y_train))
    print("R2 Score: ",r2_score(y_test, y_test_pred))
    new_y_test_pred = np.empty_like(y)
    new_y_test_pred[:] = np.nan
    new_y_test_pred[len(y_train):] = y_test_pred
    new_y_train_pred = np.empty_like(y)
    new_y_train_pred[:] = np.nan
    new_y_train_pred[:len(y_train)] = y_train_pred
    plt.figure(figsize=(14,6))
    plt.plot(y, color='b', label='Actual BTC Prices')
    plt.plot(new_y_test_pred, color='r', label='Predicted Prices')
    plt.plot(new_y_train_pred, color='g', label='Predicted Training Prices')
    plt.legend(loc='upper left')
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
    print("R2 Score: ",r2_score(y_test, y_test_pred))
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
    model = RandomForestRegressor(criterion='mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0, max_features='auto', bootstrap=False, warm_start=True)
    #print(tuned_model.cv_results_)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("Training Accuracy: ",model.score(x_train, y_train))
    print("R2 Score: ",r2_score(y_test, y_test_pred))
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
def incremental_model_daily_btc(dataset):
    #Create feature for future prices
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    prices = pd.DataFrame(dataset.drop(['Date'], axis=1).values)
    prices.columns = names
    prediction_range = 30
    test_size = 563
    #Split the data into x and y
    x = np.array(prices.drop(['Close','Adj Close'], axis=1))
    x_true = x[:len(prices) - prediction_range]
    print(x_true)
    y = np.array(prices['Close'])
    y_true = y[: -prediction_range]
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    model = HoeffdingAdaptiveTreeRegressor()
    n_samples = 0
    correct_pred = 0
    pred_prices = []
    while n_samples < x_scaled.shape[0]-30:
        if n_samples == 10:
            break
        x, y = [x_true[n_samples]], [y_true[n_samples]]
        model.partial_fit(x,y)
        pred = model.predict(x)
        if y == pred:
            correct_pred += 1
        n_samples += 1
        pred_prices.append(pred)
        print(y, pred)
    print(r2_score(y_true, pred_prices))
    plt.figure(figsize=(14,6))
    plt.plot(y_true, 'b', label='Actual Prices')
    plt.plot(pred_prices, 'r', label='Predicted Prices')
    plt.legend(loc='upper left')
    plt.show()

def incremental_model_minute_btc():
    scaler = MinMaxScaler()
    model = HoeffdingAdaptiveTreeRegressor()
    pred_prices = []
    actual_prices = []
    while True:
        x, y = collect_price_data_btc()
        #x_scaled = scaler.fit_transform(x)
        #print(scaler.scale_)
        x, y = [x[:]], [y]
        model.partial_fit(x, y)
        pred = model.predict(x)
        actual_prices.append(y)
        pred_prices.append(pred)
        print(y, pred)
        print(actual_prices, pred_prices)
        #print(r2_score(actual_prices, pred_prices))

        
