from Functions.prediction_models import *

btc_usd_data = pd.read_csv('Datasets/BTC-USD_2020-2021.csv', date_parser=True)
eth_usd_data = pd.read_csv('Datasets/ETH-USD.csv', date_parser=True)
ltc_usd_data = pd.read_csv('Datasets/LTC-USD.csv', date_parser=True)
ada_usd_data = pd.read_csv('Datasets/ADA-USD.csv', date_parser=True)
xlm_usd_data = pd.read_csv('Datasets/XLM-USD.csv', date_parser=True)
xrp_usd_data = pd.read_csv('Datasets/XRP-USD.csv', date_parser=True)
#linear_regression_btc(btc_usd_data)
#linear_regression_eth(eth_usd_data)
#linear_regression_ltc(ltc_usd_data)
#linear_regression_ada(ada_usd_data)
#linear_regression_xrp(xrp_usd_data)
#linear_regression_xlm(xlm_usd_data)
#lstm_btc(btc_usd_data)
#lstm_eth(eth_usd_data)
#lstm_ltc(ltc_usd_data)
#lstm_ada(ada_usd_data)
#lstm_xrp(xrp_usd_data)
#lstm_xlm(xlm_usd_data)
#svr_btc(btc_usd_data)
#svr_eth(eth_usd_data)
#svr_ltc(ltc_usd_data)
#svr_ada(ada_usd_data)
#svr_xrp(xrp_usd_data)
#svr_xlm(xlm_usd_data)
#neural_networks_btc(btc_usd_data)
#neural_networks_eth(eth_usd_data)
#neural_networks_ltc(ltc_usd_data)
#neural_networks_ada(ada_usd_data)
#neural_networks_xrp(xrp_usd_data)
neural_networks_xlm(xlm_usd_data)
#extra_trees_btc(btc_usd_data)
#extra_trees_eth(eth_usd_data)
#extra_trees_ltc(ltc_usd_data)
#extra_trees_ada(ada_usd_data)
#extra_trees_xrp(xrp_usd_data)
#extra_trees_xlm(xlm_usd_data)
#incremental_model_daily_btc(btc_usd_data)
#incremental_model_daily_eth(eth_usd_data)
#incremental_model_daily_ltc(ltc_usd_data)
#incremental_model_daily_ada(ada_usd_data)
#incremental_model_daily_xrp(xrp_usd_data)
#incremental_model_daily_xlm(xlm_usd_data)
