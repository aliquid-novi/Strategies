import MetaTrader5 as mt5 
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller 
from datetime import datetime
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error 
from datetime import datetime, timedelta 

scaler = MinMaxScaler()

mt5.initialize()
# Replace following with your MT5 Account Login
account=51434456 # 
password="9UpBvVzc"
server = 'ICMarkets-Demo'

import warnings
warnings.filterwarnings("ignore")

def get_rates(pair1, timeframe, x):
    pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, timeframe, 0, x))
    pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')
    return pair1[['time','open', 'high', 'low', 'close']].set_index('time')

def compute_spread(p1, p2, tf, x):
    data1 = get_rates(p1, tf, x)
    data2 = get_rates(p2, tf, x)
    merged = data1.join(data2, lsuffix="_x", rsuffix="_y")
    spread = merged['close_x'] - merged['close_y']
    return spread.dropna()

def adf_test(spread):
    '''Runs ADF test on a spread series'''
    result = adfuller(spread)
    return {'ADF Statistic': result[0], 'p-value': result[1], 'Critical Values': result[4]}

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
        "comment": 'granger_xgboost',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling':mt5.ORDER_FILLING_IOC,

        }
    result = mt5.order_send(request)

def send_order(symbol, side, lot, comment):

    if side.lower() == 'sell':
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    elif side.lower() == 'buy':
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": 5,
        "magic": 234000,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    result

def check_close_position(trade_position):
    # Assuming trade_position.time is a Unix timestamp in UTC
    time = datetime.fromtimestamp(trade_position.time)
    
    print(f"Trade opened at: {time}")
    
    current_utc_time = datetime.utcnow()
    
    # Convert to GMT+2
    current_gmt2_time = current_utc_time + timedelta(hours=2)
    
    # Calculate the time difference in seconds
    time_difference = (current_gmt2_time - time).total_seconds()
    
    # Check if 24 hours or more have passed (24 hours = 24 * 60 * 60 seconds)
    if time_difference >= 86400:
        print(f"--------------------- Closing {trade_position.symbol} of {trade_position.comment}")
        close_position(trade_position)
    else:
        hours_open = time_difference / 3600
        print(f"Position {trade_position.symbol} of {trade_position.comment} has been open for less than 24 hours. Time open: {round(hours_open, 2)} hrs")

def calc_hedge_ratio(x, y):
    # Align the series and drop NA values
    x, y = x.align(y, join='inner')
    x.dropna(inplace=True)
    y = y.loc[x.index]

    # Add a constant to the model
    y = sm.add_constant(y)

    model = sm.OLS(x, y)
    results = model.fit()
    hedge_ratio = round(results.params[1], 2)  # Assuming the constant is the first parameter
    return hedge_ratio

def get_pair_correlations(symbol1, symbol2, window, CSS_sym1, CSS_sym2):
    # css = corresponding symbol spread. Getting the yen equivalent essentially.
    s1 = str(symbol1)
    s2 = str(symbol2)
    # print(symbol1)
    symbol1 = get_rates(s1, mt5.TIMEFRAME_D1, 2000)
    symbol2 = get_rates(s2, mt5.TIMEFRAME_D1, 2000)

    combined_df = pd.concat([symbol1['close'].rename(f'{s1}_close'),
                            symbol2['close'].rename(f'{s2}_close')], axis=1)

    window_size = window  # Change this to the size of the window you want
    combined_df['rolling_corr'] = combined_df[f'{s1}_close'].rolling(window=window_size).corr(combined_df[f'{s2}_close'])
    combined_df['spread'] = combined_df[f'{s1}_close'] - combined_df[f'{s2}_close'] 
    combined_df[f'{s1}_return'] = combined_df[f'{s1}_close'].pct_change()
    combined_df[f'{s2}_return'] = combined_df[f'{s2}_close'].pct_change()
    combined_df['diff'] = combined_df[f'{s1}_return'] - combined_df[f'{s2}_return']
    combined_df['rolling_corr_returns'] = combined_df['rolling_corr'].rolling(window=window_size).corr(combined_df['diff'])
    combined_df['rolling_var'] = combined_df['spread'].rolling(window = 25).var()
    combined_df['MA_Ratio'] = combined_df['spread'].rolling(window = 15).mean() / combined_df['spread'].rolling(window = 75).mean()
    
    css1 = str(CSS_sym1)
    css2 = str(CSS_sym2)
    
    css_sym1 = get_rates(css1, mt5.TIMEFRAME_D1, 2000)
    css_sym2 = get_rates(css2, mt5.TIMEFRAME_D1, 2000)
    
    combined_df[f'{css1}_close'] = css_sym1['close']
    combined_df[f'{css2}_close'] = css_sym2['close']
    
    combined_df['CSS_spread'] = combined_df[f'{css1}_close'] - combined_df[f'{css2}_close']
    combined_df[f'{css1}_return'] = combined_df[f'{css1}_close'].pct_change()
    combined_df[f'{css2}_return'] = combined_df[f'{css2}_close'].pct_change()
    combined_df['CSS_diff'] = combined_df[f'{css1}_return'] - combined_df[f'{css2}_return']
    combined_df['CSS_rolling_corr_returns'] = combined_df['rolling_corr'].rolling(window=window_size).corr(combined_df['CSS_diff'])
    combined_df['CSS_rolling_var'] = combined_df['CSS_spread'].rolling(window = 25).var()
    combined_df['CSS_MA_Ratio'] = combined_df['CSS_spread'].rolling(window = 15).mean() / combined_df['spread'].rolling(window = 75).mean()
    
    combined_df.fillna(method = 'bfill', inplace = True)
    
    return combined_df 

def granger_causality_test(pair_df):
    
    granger_dict = {}  # Initialize an empty dictionary

    for col in pair_df.columns:
        if col != 'spread':  # Skip the 'spread' column itself
            # Create a DataFrame with only the 'spread' column and the current column
            data_subset = pair_df[['spread', col]]
            
            # Check for stationarity first
            
            adf_stat = adf_test(pair_df[col])
            
            if adf_stat['ADF Statistic'] < adf_stat['Critical Values']['10%']:
                
            
                # Perform the Granger causality test
                grang_test = grangercausalitytests(data_subset, maxlag=15, verbose=True)
                # Initialize an empty list to store significant lags for the current column
                significant_lags = []
                for i in range(1, 15):
                    score = 0

                    for test in grang_test[i][0]:
                        p_val = grang_test[i][0][test][1]
                        if p_val < 0.07:
                            score += 1 
                    if score == 4:
                        print(f"Lag {i} is significant")
                        # Append the significant lag to the list
                        significant_lags.append(i)

                if significant_lags:
                    granger_dict[col] = significant_lags
                    
            else:
                continue
    return granger_dict

def create_grang_df(original_df, granger_dict):
    
    granger_df = pd.DataFrame()
    
    print(granger_dict) # DEBUG
    
    for col, lags in granger_dict.items():
        
        for lag in lags:
            
            print(original_df) # DEBUG
            
            granger_df[f'{col}_lag{lag}'] = original_df[col].shift(lag)
                
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()

    granger_df = pd.DataFrame(scaler.fit_transform(granger_df), columns=granger_df.columns, index=granger_df.index)
    granger_df['spread'] = original_df['spread']

    return granger_df.dropna()

def create_grang_df_v2(original_df, granger_dict):
    granger_df = pd.DataFrame(index=original_df.index)
    
    for col, lags in granger_dict.items():
        for lag in lags:
            granger_df[f'{col}_lag{lag}'] = original_df[col].shift(lag)
                
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(granger_df.dropna())
    granger_df = pd.DataFrame(scaled_data, columns=granger_df.columns, index=granger_df.dropna().index)
    granger_df['spread'] = original_df['spread']

    return granger_df

tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'] 

currency_pairs = [
    ('EURUSD', 'GBPUSD', 'EURJPY', 'GBPJPY'),
    ('AUDCAD', 'USDCAD', 'CADJPY', 'AUDJPY'),
    ('NZDCAD', 'USDCAD', 'CADJPY', 'NZDJPY'),
    ('EURCHF', 'GBPCHF', 'CHFJPY', 'EURJPY'),
    # ('EURNOK', 'EURSEK', 'NOKJPY', 'SEKJPY')
]

dict_df = {}
for symbol in currency_pairs:
    dict_df[f'{symbol[0][0:]}_{symbol[1][0:]}'] = get_pair_correlations(symbol[0], symbol[1], 5, symbol[2], symbol[3]) 

# dict_df = dict_df.dropna()

combined_granger_dict = {}

for name, values in dict_df.items():
    combined_granger_dict[name] = granger_causality_test(values)
    
granger_dfs = {}

for df in dict_df.keys():

    granger_dfs[df] = create_grang_df_v2(dict_df[df], combined_granger_dict[df])
    
# import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# import yfinance as yf
import pandas_ta as ta
from sklearn.metrics import r2_score

# for df_name, df_data in granger_dfs.items():
#     print(f"Iterating through {df_name}")
#     features = df_data.columns.drop('spread')
#     target = 'spread'

#     # Create and train an XGBoost model
#     xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    
#     xgb_model.fit(df_data[features], df_data[target])

#     # Make predictions (you might want to predict on new unseen data instead)
#     prediction_data = df_data[features].iloc[-1:].copy()
#     predictions = xgb_model.predict(prediction_data)
#     current_spread = df_data[target].iloc[-1]
#     print(f"Prediction : {predictions}. Current spread: {current_spread}")

#     direction = predictions - current_spread

#     symbol1 = df_name[0:6] + '.a'
#     symbol2 = df_name[7:] + '.a'

#     sym1 = get_rates(symbol1, mt5.TIMEFRAME_D1, 2000)
#     sym2 = get_rates(symbol2, mt5.TIMEFRAME_D1, 2000)
    
#     sym1 = sym1['close']
#     sym2 = sym2['close']
    
#     print(f'Calculating hedge ratio for {symbol1} and {symbol2}')
#     ratio = calc_hedge_ratio(sym1, sym2)
#     print(f"Hedge ratio for {df_name} is {ratio}")

#     hedge_lot = abs(float(round((ratio * 0.75), 2)))

#     if direction > 0:
#         print("Buying")
#         send_order(symbol1, 'buy', 0.6, 'grang_d')
#         send_order(symbol2, 'sell', hedge_lot, 'grang_d')
#     else:
#         print("Selling")
#         send_order(symbol2, 'sell', hedge_lot, 'grang_d')
#         send_order(symbol1, 'buy', 0.6, 'grang_d')

for i in mt5.positions_get():
    if i.comment == 'grang_d':
        check_close_position(i) 
