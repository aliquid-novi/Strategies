import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
warnings.filterwarnings("ignore")
mt5.initialize()
account=51127988
password="Aar2frM7"
server = 'ICMarkets-Demo'

def get_rates(pair1, tf, x):
    pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, tf, 0, x))
    pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')
    pair1 = pair1.set_index(pair1['time'])
    pair1 = pair1.drop(columns = ['time','tick_volume', 'spread', 'real_volume'])
    return pair1

def corr_tool(symbol1, symbol2, timeframe, bars):
    print(f'Starting Pearson Correlation Analysis for {symbol1}, {symbol2} at {timeframe} for {bars} bars.')

    s1 = input(f'Start Time for {symbol1}?')
    s2 = input(f'End time for {symbol1}?')

    e1 = input(f'Start time for {symbol2}?')
    e2 = input(f'End time for {symbol2}')

    pair1 = get_rates(symbol1, timeframe, bars)
    explored_seg1 = pair1.between_time(s1, s2)
    pair2 = get_rates(symbol2,  timeframe, bars)
    explored_seg2 = pair2.between_time(e1, e2)
    
    print('Times to investigate are:'
          f'\n{s1} to {s2} and {e1} to {e2}')
    
    pair1_returns = pair1['open'] - pair1['close']
    pair2_returns = pair2['open'] - pair2['close']

    lst = []

    un_corr = []

    for i in range(50):
        pearson_corr = np.corrcoef(pair1_returns.tail(50), pair2_returns.tail(50))
        if pearson_corr[0][1] > 0.5:
            # print(f'Pearson correlation {pearson_corr[0][1]} on lag {i}')
            lst.append(i)
        else:
            un_corr.append(i)
            
    print(f'Lags that have a correlation higher than 0.5 are:'
        f'\n{lst}')
    print(f'Total Lags: {len(lst)}')

    import datetime

    day_counts = {}

    for i in lst:
        date = syd_opens.index[i]
        day_name = date.day_name()

        if day_name not in day_counts:
            day_counts[day_name] = 0

        day_counts[day_name] += 1
    print(f'Days that have correlations between {symbol1} and {symbol2}of more than 0.5 are:')

    for day, count in day_counts.items():
        print(f'{day}: {count}')
        
    day_counts = {}

    for i in un_corr:
        date = syd_opens.index[i]
        day_name = date.day_name()

        if day_name not in day_counts:
            day_counts[day_name] = 0

        day_counts[day_name] += 1
    print(f'Days that do not have correlations between {symbol1} and {symbol2} of more than 0.5 are:')

    for day, count in day_counts.items():
        print(f'{day}: {count}')