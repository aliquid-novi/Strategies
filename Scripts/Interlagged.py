def run():

    import MetaTrader5 as mt5
    import pandas as pd
    import numpy as np
    import warnings
    import pytz
    warnings.filterwarnings("ignore")
    mt5.initialize()

    account=51127988
    password="Aar2frM7"
    server = 'ICMarkets-Demo'

    from collections import defaultdict
    from datetime import datetime, timedelta

    def get_rates(pair1, tf, x):
        pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, tf, 0, x))
        pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')
        pair1 = pair1.set_index(pair1['time'])
        pair1 = pair1.drop(columns = ['time','tick_volume', 'spread', 'real_volume'])
        return pair1

    # Assume mini_list and get_rates are defined elsewhere.
    mini_list = ['AUDUSD.a', 'CADJPY.a', 'EURCAD.a', 'EURGBP.a']

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
            "comment": f"Rips Complex",
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
            "comment": f"Rips Complex",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result1 = mt5.order_send(request)
        result1

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
    # current_time2 = '20:00'
    pairs2trade = []

    for data in correct_data:
        time_range = data[0]
        start_time, end_time = [t.strftime('%H:%M') for t in [datetime.strptime(t, '%H:%M') for t in time_range.split('-')]]
        if end_time == current_time:
            print(f'Match Found for {data[2]} at {end_time}. Adding to list to trade...')
            pairs2trade.append(data)
            
        else:
            print(f'No Match on {data[2]}. Time is {current_time}. Correlated time is {end_time}')

    def get_rate(pair1, tf, x):
        pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, tf, 0, x))
        pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')
        pair1 = pair1.set_index(pair1['time'])
        pair1 = pair1.drop(columns = ['time','tick_volume', 'spread', 'real_volume'])
        return pair1

    from datetime import time
    buy_list = []
    sell_list = []

    for pair in pairs2trade:
        rates = get_rate(pair[2], mt5.TIMEFRAME_H1, 24)
        rates.index = rates.index.time
        target_time = time(20,0,0)
        info = rates.loc[target_time]
        returns = (info['close'] - info['open'])
        print(f"Returns for {pair[2]} are {returns}")

        if returns > 0:
            print(f"Init. Buy for {pair}")
            buy_list.append(pair)
            
        else:
            print(f"Init. Sell for {pair}")
            sell_list.append(pair)
        
    print(buy_list)
    print(sell_list)

    import datetime

    # Store position opening times
    open_positions = {}

    # ... code to open position ...

    # After opening a position, store the opening time
    for position in mt5.positions_get():
        open_positions[position.ticket] = datetime.datetime.now()

    # ... main loop ...

    # Iterate through open positions
    if len(open_positions) >= 1:
        print('Beginning closing logic')
        for ticket, open_time in open_positions.items():
            # Get the current time
            current_time = datetime.datetime.now()
            # Calculate the difference between the current time and the open time
            time_difference = current_time - open_time
            # If the position has been open for more than an hour
            if time_difference.total_seconds() > 3600:  # 3600 seconds = 1 hour
                # Close the position
                position = mt5.positions_get(ticket=ticket)[0]
                close_position(position)
                # Remove the position from the dictionary
                del open_positions[ticket]
    else:
        print('No open positions') 