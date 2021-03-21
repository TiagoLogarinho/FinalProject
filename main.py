from Functions.prediction_models import *

btc_usd_data = pd.read_csv('Datasets/BTC-USD_2020-2021.csv', date_parser=True)
btc_price_data = pd.read_csv('Datasets/archive/bitcoin_price.csv')
#linear_regression_btc(btc_usd_data)
#Linear Regression now achieves a much more realistic result, of 71% accuracy when trying to predict prices 30 days in the future
#lstm_btc(btc_usd_data)
#svr_btc(btc_usd_data)
#neural_networks_btc(btc_usd_data)
#SVR is causing issues with the same implementation as LR
##NN is now fixed, as the "best hyperparameters" have been selected to run the model
#Need to solve it tomorrow or wait until wednesday to show Diana
#random_forest_btc(btc_usd_data)
#incremental_model_daily_btc(btc_usd_data)
incremental_model_minute_btc()