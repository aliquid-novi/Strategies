import MetaTrader5 as mt5 
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
mt5.initialize()
# Replace following with your MT5 Account Login
account=51434456 # 
password="51469692"
server = 'ICMarkets-Demo'
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
        "comment": 'granger_arima',
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
    
EU_GU = get_pair_correlations('EURUSD.a', 'GBPUSD.a', 5, 'EURJPY.a', 'GBPJPY.a')

granger_data = granger_causality_test(EU_GU)
granger_results = create_grang_df_v2(EU_GU, granger_data)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df = granger_results

import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# Assuming 'df' is the DataFrame with your data

# Split the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# ARIMAX Model - using p, d, q values as 1 for demonstration; these should be determined by analysis
# Extracting the target variable and the features
y_train = train_df['spread']
X_train = train_df.drop('spread', axis=1)
y_test = test_df['spread']
X_test = test_df.drop('spread', axis=1)

# Building the ARIMAX model
arimax_model = SARIMAX(y_train, exog=X_train, order=(1, 0, 1))
arimax_result = arimax_model.fit()

# Making predictions
arimax_predictions = arimax_result.get_forecast(steps=len(y_test), exog=X_test).predicted_mean
arimax_mse = mean_squared_error(y_test, arimax_predictions)


arimax_predictions = arimax_result.get_forecast(steps=len(y_test), exog=X_test).predicted_mean
forecast = arimax_predictions.iloc[-1]
forecast

def current_spread(symbol1, symbol2, tf):
    
    s1 = get_rates(symbol1, tf, 1)
    s2 = get_rates(symbol2, tf, 1)
    
    return float(s1['close'] - s2['close'])

current_price = current_spread('EURUSD.a', 'GBPUSD.a', mt5.TIMEFRAME_H4)
current_price

if current_price > forecast:
    print("Buying EURUSD, selling GBPUSD")
    print(f"Current Price: {current_price}. Current forecast: {forecast}")
    send_order('EURUSD.a', 'buy', 1.00, 'grang_arima')
    send_order('GBPUSD.a', 'sell', 0.87, 'grang_arima')
else:
    print("Buying GBPUSD, selling EURUSD")
    print(f"Current Price: {current_price}. Current forecast: {forecast}")
    send_order('EURUSD.a', 'sell', 1.00, 'grang_arima')
    send_order('GBPUSD.a', 'buy', 0.87, 'grang_arima')

def check_close_position(trade_position):

    from datetime import datetime
    from datetime import datetime, timedelta
    # Your logic to close the position
    ticket = trade_position.ticket
    time = datetime.fromtimestamp(trade_position.time)
    print(time)
    symbol = trade_position.symbol
    comment = trade_position.comment
    # print(f"Checking {symbol} of {comment} to close")
    # Sample trade position (replace this with the actual object you get)
    position = {
        'ticket': ticket,
        'time': time,  # This would be the actual Unix timestamp
        'Symbol': symbol,
        'Comment': comment,
    }

    local_to_utc_offset = timedelta(hours= +3)

    # Get the current time in GMT+3
    current_time_gmt_plus_3 = datetime.now() 

    # Convert it to UTC
    current_time_utc = current_time_gmt_plus_3 + local_to_utc_offset

    # Convert to Unix timestamp
    current_unix_time = int(current_time_utc.timestamp())

    # Calculate the time difference in seconds
    time_difference = current_unix_time - int(i.time)
    
    # Check if 4 hours or more have passed (4 hours = 4 * 60 * 60 seconds)
    if time_difference >= 14400:
        print(f"Time difference is {round((time_difference / 60),2) / 60} hrs. Closing {symbol} of {comment}")
        close_position(trade_position)
    else:
        print(f"Position {symbol} of {comment} has been open for less than 4 hours. Time open: {round(round((time_difference / 60),2) / 60), 4} hrs")

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

for i in mt5.positions_get():
    if i.comment == 'grang_arima':
        check_close_position(i)