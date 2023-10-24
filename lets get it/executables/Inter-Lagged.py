import MetaTrader5 as mt5
from datetime import datetime
import pytz
import pandas as pd
import numpy as np


mt5.initialize()
account=51434456 # 
password="9UpBvVzc"
server = 'ICMarkets-Demo'

from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def run():
    def get_rates(pair1, tf, x):
        pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, tf, 0, x))
        pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')
        pair1 = pair1.set_index(pair1['time'])
        pair1 = pair1.drop(columns = ['time','tick_volume', 'spread', 'real_volume'])
        return pair1

    # Assume mini_list and get_rates are defined elsewhere.
    mini_list = ['AUDUSD.a', 'CADJPY.a', 'EURCAD.a', 'EURGBP.a', 'EURUSD.a', 'GBPUSD.a', 'AUDJPY.a', 'USDJPY.a', 'NZDUSD.a', 'USDCHF.a', 'GBPAUD.a']

    def buy_order(symbol):
        price = mt5.symbol_info_tick(symbol).ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 1.00,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Interlagged_B",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result1 = mt5.order_send(request)
        result1

    def sell_order(symbol):
        price = mt5.symbol_info_tick(symbol).bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 1.00,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Interlagged_S",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result1 = mt5.order_send(request)
        result1

    def get_rate(pair1, tf, x):
        pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, tf, 0, x))
        pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')
        pair1 = pair1.set_index(pair1['time'])
        pair1 = pair1.drop(columns = ['time','tick_volume', 'spread', 'real_volume'])
        return pair1
    
    def t3_sing_corr(symbol, base_hour, bars=100):
        base_ts1 = datetime.strptime(f'{base_hour}:00', '%H:%M').time()
        base_ts2 = (datetime.combine(datetime.today(), base_ts1) + timedelta(hours=1)).time()
        base_ts1, base_ts2 = [t.strftime('%H:%M') for t in [base_ts1, base_ts2]]

        base_prices = get_rates(symbol, mt5.TIMEFRAME_H1, 50000)
        base_prices = base_prices.between_time(base_ts1, base_ts2)
        base_prices['returns'] = base_prices['close'] - base_prices['open']

        correlations = {}
        ordered_correlations = defaultdict(list)

        for h in range(24):
            if h == base_hour:
                continue

            ts1 = datetime.strptime(f'{h}:00', '%H:%M').time()
            ts2 = (datetime.combine(datetime.today(), ts1) + timedelta(hours=1)).time()
            ts1, ts2 = [t.strftime('%H:%M') for t in [ts1, ts2]]

            prices = get_rates(symbol, mt5.TIMEFRAME_H1, 50000)
            prices = prices.between_time(ts1, ts2)
            prices['returns'] = prices['close'] - prices['open']

            pearson_corr = np.corrcoef(base_prices['returns'].tail(bars), prices['returns'].tail(bars))[0, 1]

            if abs(pearson_corr) > 0.2:
                correlations[f'{base_ts1}-{ts1} | {symbol}'] = pearson_corr

            if abs(pearson_corr) > 0.4:
                ordered_correlations['0.4 and over'].append((f'{base_ts1}-{ts1}', pearson_corr, symbol))
            elif abs(pearson_corr) > 0.3:
                ordered_correlations['0.3 to 0.4'].append((f'{base_ts1}-{ts1}', pearson_corr, symbol))
            elif abs(pearson_corr) > 0.2:
                ordered_correlations['0.2 to 0.3'].append((f'{base_ts1}-{ts1}', pearson_corr, symbol))

        return correlations, ordered_correlations

    correlations_all = {}
    ordered_correlations_all = defaultdict(list)

    for symbol in mini_list:
        for i in range(24):
            correlations, ordered_correlations = t3_sing_corr(symbol, i, bars=100)
            correlations_all.update(correlations)
            for key in ordered_correlations.keys():
                ordered_correlations_all[key].extend(ordered_correlations[key])

    print('Analysis Complete')

    data = ordered_correlations_all
    ordered_correlations_all = defaultdict(list)
    for key in data.keys():
        ordered_correlations_all[key] = [item for item in data[key] if int(item[0].split('-')[0].split(':')[0]) < int(item[0].split('-')[1].split(':')[0])]

    strong_correlations = ordered_correlations_all['0.4 and over']
    correct_data = [item for item in strong_correlations if int(item[0].split('-')[0].split(':')[0]) < int(item[0].split('-')[1].split(':')[0])]
    print('Sorting complete. Data:')
    print(correct_data)

    now = datetime.now(pytz.timezone('Europe/Helsinki'))  # use the appropriate timezone
    current_time = now.strftime('%H:%M')
    pairs2trade = []

    for data in correct_data:
        time_range = data[0]
        start_time, end_time = [datetime.strptime(t, '%H:%M') for t in time_range.split('-')]
        end_time = end_time.replace(year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo)
        current_time_dt = datetime.strptime(current_time, '%H:%M').replace(year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo)

        time_difference = abs((end_time - current_time_dt).total_seconds())

        if time_difference <= 500:  # Within 5 minutes
            print(f'Match Found for {data[2]} at {end_time.strftime("%H:%M")}. Adding to list to trade...')
            pairs2trade.append(data)
        else:
            print(f'No Match on {data[2]}. Time is {current_time}. Correlated time is {end_time.strftime("%H:%M")}')

    print(pairs2trade)
    
    # Send Orders #   
    for pair in pairs2trade:
        end_time_str = pair[0].split('-')[1]
        target_time = datetime.strptime(end_time_str, '%H:%M').time()
        rates = get_rate(pair[2], mt5.TIMEFRAME_H1, 24)
        rates.index = rates.index.time
        info = rates.loc[target_time]  # use the dynamically set target_time
        returns = (info['close'] - info['open'])

        if returns > 0:
            print(f"Init. Buy for {pair[2]}")
            buy_order(pair[2])

        else:
            print(f"Init. Sell for {pair[2]}")
            sell_order(pair[2])

    from datetime import time

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

    def check_close_position(trade_position):
        from datetime import datetime
        from datetime import datetime, timedelta
        # Your logic to close the position
        ticket = trade_position.ticket
        time = datetime.fromtimestamp(trade_position.time)
        symbol = trade_position.symbol
        comment = trade_position.comment
        print(f"Checking {symbol} of {comment} to close")
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
            close_position(trade_position)
        else:
            print(f"Position {symbol} of {comment} has been open for less than 1 hours.")

    open_positions = mt5.positions_get()

    for i in open_positions:
        # print(i)
        if 'Interlagged' in i.comment:
            check_close_position(i)

if __name__ == "__main__":
    run()