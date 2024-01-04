import MetaTrader5 as mt5 
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller 
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings("ignore")
mt5.initialize()
# Replace following with your MT5 Account Login
account = 533690
password = 'gywNZ76##' 
server = 'GrowthNext-Server'

def get_rates(pair1, timeframe, x):
    pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, timeframe, 0, x))
    pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')
    return pair1[['time','open', 'high', 'low', 'close']].set_index('time')

def calc_stats(df):

    stats_df = pd.DataFrame()
    stats_df['mean'] = df.mean()
    stats_df['std'] = df.std()
    # print('calculated')
    return stats_df



