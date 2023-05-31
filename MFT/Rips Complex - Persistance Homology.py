import MetaTrader5 as mt5
from datetime import datetime
import pytz
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
from sklearn.metrics.pairwise import euclidean_distances
import gudhi as gd
import shutil
warnings.filterwarnings("ignore")
mt5.initialize()
account=51127988
password="Aar2frM7"
server = 'ICMarkets-Demo'

def get_rates(pair1, x):
    pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, mt5.TIMEFRAME_M15, 0, x))
    pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')
    return pair1

AUDUSD = get_rates('AUDUSD.a', 150)

df = AUDUSD[['open', 'high', 'low', 'close']]

# Compute the Euclidean distance matrix
dist_matrix = euclidean_distances(df)

# Use the GUDHI library to construct a Rips complex
rips_complex = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=1.0)

simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

# To visualize or analyze the topological features of the complex, you can use GUDHI's persistence diagram functionality:
diag = simplex_tree.persistence(min_persistence=0.01)

# Set a threshold for the minimum persistence that we consider significant
min_persistence = 0.1
threshold_high = 20
threshold_low = 5

# Calculate the persistence diagram
diag = simplex_tree.persistence(min_persistence=min_persistence)

# Extract the persistent features (those with persistence above the threshold)
persistent_features = [interval for interval in diag if interval[1][1] - interval[1][0] > min_persistence]

# Hypothetical trading logic
if len(persistent_features) > threshold_high:
    print("The market is complex, potentially indicating a trend change. Consider selling if in a long position.")
    if len(mt5.positions_get()) == 0:
        buy_order()
    else:
        for i in mt5.positions_get():
            close_position(i)
elif len(persistent_features) < threshold_low:
    print("The market is simple, potentially indicating a trend continuation. Consider buying if in a short position.")
    if len(mt5.positions_get()) == 0:
        sell_order()
    else:
        for i in mt5.positions_get():
            close_position(i)
else:
    print("The market is in an intermediate state. No action taken.")
    
def buy_order():
    price = mt5.symbol_info_tick('AUDUSD.a').ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": 'AUDUSD.a',
        "volume": 1.00,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": f"Rips Complex",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result1 = mt5.order_send(request)
    result1
    
def sell_order():
    price = mt5.symbol_info_tick('AUDUSD.a').bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": 'AUDUSD.a',
        "volume": 1.00,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": f"Rips Complex",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result1 = mt5.order_send(request)
    result1
    
def close_position(position):

    tick = mt5.symbol_info_tick(position.symbol)

    request = {
        "action" : mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
        "price": tick.ask if position.type == 1 else tick.bid,
        "deviation": 20,
        "magic": 100,
        "comment": 'pytohn script close',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling':mt5.ORDER_FILLING_IOC,

        }
    result1 = mt5.order_send(request)