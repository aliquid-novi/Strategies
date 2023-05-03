#!/usr/bin/env python
# coding: utf-8

# In[1]:


import MetaTrader5 as mt5
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import adfuller
mt5.initialize()
account=51127988
password="Aar2frM7"
server = 'ICMarkets-Demo'
if not mt5.initialize(login=account, password=password, server=server):
    print("initialize() failed, error code =",mt5.last_error())
    quit()
authorized=mt5.login(account, password=password, server=server)

if authorized:
    print('Authorized')
else:
    print("failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))


# ## 7-period 2min Trading Hypothesis 
# Following prices movements on the 7-period MA on the 2min time interval for every bar that has a range of more than 3 pips will present opportunities to follow the momentum of the market

# ### Data Collection

# In[2]:

def run():

    def get_rates(pair1, x, tf):
        pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, tf, 0, x))
        pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')
        pair1 = pair1[['time', 'open', 'high', 'low', 'close']]
        return pair1 


    symbols = ['AUDUSD.a', 'EURUSD.a', 'GBPUSD.a', 'NZDUSD.a', 'GBPJPY.a']


    # ### Trading Logic

    # In[4]:

    # In[5]:


    def short_order():
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(i).bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": i,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": 234000,
            "comment": f"ST_7P_2M",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            }
        result = mt5.order_send(request)
        result
    def long_order():
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(i).ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": i,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": 234000,
            "comment": f"ST_7P_2M",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            }
        result = mt5.order_send(request)


    # In[11]:


    lot = 1.00
    deviation = 5

    for symbol in symbols:
        data = get_rates(symbol, 360, mt5.TIMEFRAME_M2)
        data['7-MA'] = data['close'].rolling(window = 7).mean()
        
        if data['7-MA'].iloc[-1] > data['close'].iloc[-1]:
            print(f'{symbol} is a short')
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
            point = mt5.symbol_info(symbol).point
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": price - 20 * point,
                "tp": price + 30 * point,
                "deviation": deviation,
                "magic": 234000,
                "comment": f"ST_7P_2M",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
                }
            result = mt5.order_send(request)
            result
            print("Sell {} {} lots at {} with deviation={} points".format(symbol,lot,price,deviation));
        else:
            print(f'{symbol} is a long')
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
            point = mt5.symbol_info(symbol).point 
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": price - 20 * point,
                "tp": price + 30 * point,
                "deviation": deviation,
                "magic": 234000,
                "comment": f"ST_7P_2M",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
                }
            result = mt5.order_send(request)
            print("Sell {} {} lots at {} with deviation={} points".format(symbol,lot,price,deviation));



