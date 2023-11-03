import MetaTrader5 as mt5 
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller 
from datetime import datetime
mt5.initialize()
# Replace following with your MT5 Account Login
account=51434456 # 
password="9UpBvVzc"
server = 'ICMarkets-Demo'

# def run():

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

def get_pair_correlations(symbol1, symbol2, window):
    s1 = str(symbol1)
    s2 = str(symbol2)
    symbol1 = get_rates(symbol1, mt5.TIMEFRAME_H1, 10000)
    symbol2 = get_rates(symbol2, mt5.TIMEFRAME_H1, 10000)

    combined_df = pd.concat([symbol1['close'].rename(f'{s1}_close'),
                            symbol2['close'].rename(f'{s2}_close')], axis=1)

    window_size = window  # Change this to the size of the window you want
    combined_df['rolling_corr'] = combined_df[f'{s1}_close'].rolling(window=window_size).corr(combined_df[f'{s2}_close'])
    # combined_df['rolling_corr'].iloc[0:200].plot()
    combined_df[f'{s1}_return'] = combined_df[f'{s1}_close'].pct_change()
    combined_df[f'{s2}_return'] = combined_df[f'{s2}_close'].pct_change()
    combined_df['diff'] = combined_df[f'{s1}_return'] - combined_df[f'{s2}_return']
    combined_df['rolling_corr_returns'] = combined_df['rolling_corr'].rolling(window=window_size).corr(combined_df['diff'])
    combined_df['shifted_rolling_corr_returns'] = combined_df['rolling_corr_returns'].shift(1)
    return combined_df.dropna()

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
        "comment": 'Regres Close',
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
    if time_difference >= 60 * 60:
        print(f"Time difference is {round((time_difference / 60),2)} minutes. Closing {symbol} of {comment}")
        close_position(trade_position)
    else:
        print(f"Position {symbol} of {comment} has been open for less than 1 hours. Time open: {round((time_difference / 60),2), 4} minutes")

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

def arima_correl(pair1, pair2):
    df = get_pair_correlations(pair1, pair2, 5)
    
    lst = ['diff', f'{pair1}' + '_return']
    for heading in lst:
        model_diff = ARIMA(df[heading], order=(2,0,0), exog=df['shifted_rolling_corr_returns'])
        model_diff_fit = model_diff.fit()
        yhat_diff = model_diff_fit.forecast(steps=1, exog=np.array([[df['shifted_rolling_corr_returns'].iloc[-2]]]))
        
        if heading == 'diff':
            lot = 1.25

            if yhat_diff.values > 0.00003:
                send_order(pair1, 'sell', lot, 'H1/CORARIMA ' + heading[0] + heading[3])
                send_order(pair2, 'buy', lot, 'H1/CORARIMA '+ heading[0] + heading[3])
            elif yhat_diff.values < 0.00003:
                send_order(pair1, 'buy', lot, 'H1/CORARIMA '+ heading[0] + heading[3])
                send_order(pair2, 'sell', lot, 'H1/CORARIMA' + heading[0] + heading[3])
        else:
            if yhat_diff.values > 0.00003:
                send_order(pair1, 'sell', lot, 'H1/CORARIMA ' + heading[0] + heading[3])
                send_order(pair2, 'buy', lot, 'H1/CORARIMA '+ heading[0] + heading[3])
            elif yhat_diff.values < 0.00003:
                send_order(pair1, 'buy', lot, 'H1/CORARIMA '+ heading[0] + heading[3])
                send_order(pair2, 'sell', lot, 'H1/CORARIMA' + heading[0] + heading[3])

pairs = [['EURUSD.a', 'GBPUSD.a'], ['EURUSD.a', 'AUDUSD.a']]

for pair in pairs:
    arima_correl(pair[0], pair[1])

positions = mt5.positions_get()
for i in positions:
    if 'H1' in i.comment:
        print(f"Checking {i.symbol}")
        check_close_position(i)