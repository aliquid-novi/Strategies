#!/usr/bin/env python
# coding: utf-8

# In[2]:

def run():
    import re
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


    # In[3]:


    import pytz
    aest = pytz.timezone('Australia/Sydney')
    now = datetime.now(tz = aest).strftime("%I:%M%p %Z")
    print(f"Base pairs Cointegrating Mean Reversion Series executed at {now}")


    # ## Functions
    def hedge_order():
        if len(filtered_again) > 0:
            for line in filtered_again:
                pattern = r"(Buy|Sell)\s+(\w+)\s+-\s+(\w+)\s+at\s+timeframe\s+(\d+)\s+right now\s+Hedge Ratio:\s+(\d+\.\d+)"
                counter = 0
                pairs = []
                match = re.match(pattern, line)
                if match:
                    side = match.group(1)
                    pair_1 = match.group(2)
                    pair_2 = match.group(3)
                    tf = int(match.group(4))
                    hr_ratio = float(match.group(5))
                    pairs.append((side, pair_1, pair_2, tf, hr_ratio))
                    print(f'Hedging {pair_1} & {pair_2} Entering loop...')
                    if pair_1 or pair_2 == info: 
                        print('Hedging Orders Sending')

                        hr_ratio = float(match.group(5))
                        lot = 0.15
                        lot2 = float(round(lot * hr_ratio,2))
                        symbol = pair_1 
                        symbol2 = pair_2
                        deviation = 5

                        if side == 'Sell':
                            order_type = mt5.ORDER_TYPE_BUY
                            price = mt5.symbol_info_tick(symbol).ask
                            price2 = mt5.symbol_info_tick(symbol).ask

                        elif side == 'Buy':
                            order_type = mt5.ORDER_TYPE_SELL
                            price = mt5.symbol_info_tick(symbol2).bid
                            price2 = mt5.symbol_info_tick(symbol2).bid

                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": lot,
                            "type": order_type,
                            "price": price,
                            "deviation": deviation,
                            "magic": 234000,
                            "comment": f"H{pair_1}@{tf}{side}",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result1 = mt5.order_send(request)
                        result1

                        counter += 1
                        print("Hedging Order {} for {}. {} lots at {} with points at timeframe {}".format(counter, symbol, lot, price, tf))
                        request2 = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol2,
                            "volume": lot2,
                            "type": order_type,
                            "price": price2,
                            "deviation": deviation,
                            "magic": 234000,
                            "comment": f"H{pair_2}@{tf}{side}",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result2 = mt5.order_send(request2)
                        result2

                        counter += 1
                        print("Hedging Order {} for {}. {} lots at {} with points at timeframe {}".format(counter, symbol2, lot2, price2,tf))

                        break

        else:
            print("No Over Exposed Symbols/No New Signals")
    def get_exp(OE):
        OE = {}
        open_positions = mt5.positions_get()
        from collections import defaultdict
        net_exposure = defaultdict(float)
        for position in open_positions:
            base_currency = position.symbol[:3]
            quote_currency = position.symbol[3:6]
            volume = position.volume
            side = "Buy" if "Buy" in position.comment else "Sell"
            position_type = position.type

            if side == "Buy":
                if position_type == mt5.POSITION_TYPE_BUY:
                    net_exposure[base_currency] += volume
                    net_exposure[quote_currency] -= volume
                else:
                    net_exposure[base_currency] -= volume
                    net_exposure[quote_currency] += volume
            else:
                if position_type == mt5.POSITION_TYPE_SELL:
                    net_exposure[base_currency] -= volume
                    net_exposure[quote_currency] += volume
                else:
                    net_exposure[base_currency] += volume
                    net_exposure[quote_currency] -= volume

        for currency, exposure in net_exposure.items():
            print(f"{currency}: {exposure:.2f} lots")

            if abs(exposure) > 0.75:
                print(f'{currency} is over exposed.')
                OE[currency] = exposure
        return OE
    def hedge_close_logic():
        import ta
        hedge_timeframe_dict = {}
        open_positions = mt5.positions_get()
        if len(hedge_timeframe_dict) > 0:
            for timeframe, pairs in hedge_timeframe_dict.items():
                if timeframe == 16385:# CHANGE THIS TO CORRESPONDING SCRIPT
                    mt5_timeframe = mt5.TIMEFRAME_H1# CHANGE THIS TO CORRESPONDING SCRIPT
                num_points = get_num_points(mt5_timeframe)

                for i in range(0, len(pairs) -1, 2):
                    symbol_1, symbol_2 = pairs[i], pairs[i+1]
                    rates_1 = mt5.copy_rates_from_pos(symbol_1, mt5_timeframe, 0, num_points)
                    rates_2 = mt5.copy_rates_from_pos(symbol_2, mt5_timeframe, 0, num_points)
                    if len(open_positions) == 0:
                        print('No Open Positions')
                        break

                    if rates_1 is not None and rates_2 is not None:
                        df_1 = pd.DataFrame(rates_1)
                        df_1['time'] = pd.to_datetime(df_1['time'], unit='s')
                        df_2 = pd.DataFrame(rates_2)
                        df_2['time'] = pd.to_datetime(df_2['time'], unit='s')

                        merged_df = pd.merge(df_1, df_2, on='time', suffixes=(f'_{symbol_1}', f'_{symbol_2}'))
                        merged_df = merged_df.T.drop_duplicates().T
                        df1 = pd.DataFrame(merged_df[f'close_{symbol_1}'])
                        df2 = pd.DataFrame(merged_df[f'close_{symbol_2}'])
                        test2 = pd.DataFrame(df1[f'close_{symbol_1}'] - df2[f'close_{symbol_2}'])
                        test2['rsi'] = ta.momentum.RSIIndicator(close=test2[0], window=14).rsi()

                        print(f'Running Closing Logic on {symbol_1} and {symbol_2} at {timeframe}min')

                        for position in open_positions:
                            if f'{position.symbol}' in [symbol_1, symbol_2] and f'H{position.symbol}@{timeframe}' in position.comment:
                                if 'Se' in position.comment or position.comment[-1] == 'S':

                                    pip_difference = test2['rsi'].iloc[-1]
                                    if 'JPY' or 'jpy' in position.comment:
                                        pip_difference = pip_difference / 1000
                                    print(f'RSI for {symbol_1} - {symbol_2} is currently',test2['rsi'].iloc[-1])
                                    if test2['rsi'].iloc[-1] < 30:
                                        close_positions.append(position)
                                        print(f"{position.comment} to be closed")
                                    else:
                                        print(f'Keep {position.comment} open')


                                elif 'Bu' in position.comment or position.comment[-1] == 'B':


                                    pip_difference =  test2['rsi'].iloc[-1] - 70
                                    if 'JPY' or 'jpy' in position.comment:
                                        pip_difference = pip_difference / 1000
                                    print(f'{symbol_1} - {symbol_2} is currently', pip_difference, 'pips away from the 70 RSI Threshold')

                                    if test2['rsi'].iloc[-1] > 70:
                                        close_positions.append(position)
                                        print(f"{position.comment} to be closed")
                                    else:
                                        print(f'Keep {position.comment} open')
        else:
            print("No Hedged Positions")

    def split_2(pair1, pair2, x, name1, name2, t_f):
        pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, t_f, 0, x))
        pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')

        pair2 = pd.DataFrame(mt5.copy_rates_from_pos(pair2, t_f, 0, x))
        pair2['time'] = pd.to_datetime(pair1['time'], unit = 's')
        
        currency_regex = re.compile(r'\b[A-Z]{3}(JPY|jpy)\b')
        name1_has_jpy = bool(currency_regex.search(name1))
        name2_has_jpy = bool(currency_regex.search(name2))
        
        if name1_has_jpy or name2_has_jpy:
            if name1_has_jpy and name2_has_jpy:
                pass
            else:
                pair1['close'] = (pair1['close'] - pair1['close'].mean()) / pair1['close'].std()
                pair2['close'] = (pair2['close'] - pair2['close'].mean()) / pair2['close'].std()
                
                df = pd.DataFrame(pair1['close'] - pair2['close'])
                df = df.set_axis([f'{name1} - {name2}'], axis=1)
                
                return df
        else:
            df = pd.DataFrame(pair1['close'] - pair2['close'])
            df = df.set_axis([f'{name1} - {name2}'], axis=1)

            return df
    def bollinger_bands(df, column, window=14, std_devs=2):
        sma = df[column].rolling(window=window).mean()
        std_dev = df[column].rolling(window=window).std()
        upper_band = sma + (std_devs * std_dev)
        lower_band = sma - (std_devs * std_dev)
        return sma, upper_band, lower_band

    def calc_hedge_ratio(x, y):
        Model2 = sm.OLS(x['close'], y['close'])
        Hedge_Ratio2 = Model2.fit()
        hedge_ratio_float2 = round(Hedge_Ratio2.params[0], 2)
        return hedge_ratio_float2

    ##Functions
    def get_base_columns(df):
        return [col for col in df.columns if not col.startswith(('Mean', 'Upper', 'Lower'))]

    def bollinger_band(df, column, window=20, std_devs=2):
        sma = df[column].rolling(window=window).mean()
        std_dev = df[column].rolling(window=window).std()
        upper_band = sma + (std_devs * std_dev)
        lower_band = sma - (std_devs * std_dev)
        return sma, upper_band, lower_band
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
        result = mt5.order_send(request)

        # ## Data Processing

        # In[8]:

    first_order_pairs = [('AUDJPY.a', 'EURJPY.a'), ('AUDJPY.a', 'GBPJPY.a'), ('EURJPY.a', 'GBPJPY.a'),                      
                        ('USDJPY.a', 'AUDJPY.a'), ('USDJPY.a', 'GBPJPY.a'), ('USDJPY.a', 'EURJPY.a'),                     
                        ('NZDJPY.a', 'AUDJPY.a'), ('NZDJPY.a', 'GBPJPY.a'), ('NZDJPY.a', 'EURJPY.a'), ('USDJPY.a', 'NZDJPY.a'),                      
                        ('AUDUSD.a', 'EURUSD.a'), ('AUDUSD.a', 'NZDUSD.a'), ('AUDUSD.a', 'GBPUSD.a'),                      
                        ('EURUSD.a', 'NZDUSD.a'), ('EURUSD.a', 'GBPUSD.a'), ('GBPUSD.a', 'NZDUSD.a'), 
                        ('AUDJPY.a', 'AUDUSD.a'), ('AUDUSD.a', 'AUDNZD.a'), ('EURUSD.a', 'EURJPY.a'),
                        ('EURGBP.a', 'GBPUSD.a'), ('GBPCHF.a', 'GBPUSD.a'), ('GBPCAD.a', 'GBPAUD.a'), 
                        ('CADJPY.a', 'AUDJPY.a'), ('CADJPY.a', 'NZDJPY.a'), ('EURNZD.a', 'GBPNZD.a')]

    timeframes = [mt5.TIMEFRAME_H1]
    coint_dfs = []
    
    # Define the number of bars to download for each timeframe
    bars_dict = {mt5.TIMEFRAME_H1: 240}

    for tf in timeframes:
        if tf in bars_dict:
            bars = bars_dict[tf]
            dfs = [split_2(pair[0], pair[1], bars, pair[0], pair[1], tf) for pair in first_order_pairs]
            concatenated_df = pd.concat(dfs, axis=1).reset_index(drop=True)
            new_list = []
            first_order_adf_matrix = pd.DataFrame(columns=concatenated_df.columns)
            for col in concatenated_df.columns:
                result = adfuller(concatenated_df[col])
                first_order_adf_matrix.loc['ADF Statistic', col] = result[0]
                first_order_adf_matrix.loc['p-value', col] = result[1]
                first_order_adf_matrix.loc['Critical Values', col] = float(result[4]['10%'])

            for col_idx in range(first_order_adf_matrix.shape[1]):
                if first_order_adf_matrix.iloc[2, col_idx] > first_order_adf_matrix.iloc[0, col_idx]:
                    new_list.append(first_order_adf_matrix.columns[col_idx])

            first_order_coint = pd.DataFrame([concatenated_df.iloc[:, col_idx] for col_idx in range(first_order_adf_matrix.shape[1]) if first_order_adf_matrix.iloc[2, col_idx] > first_order_adf_matrix.iloc[0, col_idx]]).transpose()
            coint_dfs.append(first_order_coint)


    # # First Order Trading Loop
    # ### Data Printing


    [df1] = coint_dfs
    dict_1 = coint_dfs
    timeframes = ['H1']
    dfs = {tf: pd.DataFrame() for tf in timeframes}

    for key, df in zip(dfs.keys(), coint_dfs):
        dfs[key] = df


    aest_tz = pytz.timezone('Australia/Sydney')
    now = datetime.now(aest).strftime("%I:%M%p %Z")
    for i, df in enumerate(coint_dfs):
        print(f"Number of columns in df{i+1}: {len(df.columns)}")
    total_columns = sum([len(df.columns) for df in coint_dfs])
    print(f"Total number of cointegrating pairs: {total_columns}")
    print(f"Loop executed at {now}")

    def get_signals(df, column, index, timeframe, bars):
        sma, upper_band, lower_band = bollinger_band(df, column)
        pair1, pair2 = column.split(' - ')

        signal = ""
        symbol = pair1
        symbol_2 = pair2

        x = pd.DataFrame(mt5.copy_rates_from_pos(pair1, timeframe, 0, bars))
        x['time'] = pd.to_datetime(x['time'], unit='s')
        y = pd.DataFrame(mt5.copy_rates_from_pos(pair2, timeframe, 0, bars))
        y['time'] = pd.to_datetime(y['time'], unit='s')

        # Calculate the hedge ratio
        hr = calc_hedge_ratio(x, y)
        signal += f"\nHedge Ratio: {hr}"

        if df[column].iloc[-1] < lower_band.iloc[-1]:
            signal = f"Buy {column} at timeframe {timeframe}  right now Hedge Ratio: {hr}"
            return signal

        elif df[column].iloc[-1] > upper_band.iloc[-1]:
            signal = f"Sell {column} at timeframe {timeframe} right now Hedge Ratio: {hr}"
            return signal

    signals = []

    # Loop through each dataframe
    for i, df in enumerate(coint_dfs):
        base_columns = get_base_columns(df)
        base_data = df[base_columns]

        # Apply trading logic
        for timeframe in bars_dict.keys():
            bars = bars_dict[timeframe]

            # Apply trading logic for each column
            for column in base_data.columns:
                signal = get_signals(base_data, column, i, timeframe, bars)
                if signal and signal not in signals:
                    signals.append(signal)

    # Extract unique signals
    signals = list(set(signals))

    # Print the signals
    for signal in signals:
        if signal is not None:
            print(signal)

    buy_counter = 0
    sell_counter = 0
    for signal in signals:
        if signal is not None and signal[0:3] == 'Buy':
            buy_counter += 1
        else:
            sell_counter += 1

    print(' Total Signals:', len(signals))
    print(' Buy Signals:', buy_counter, 'Sell Signals:', sell_counter)


    # ### Send Orders
    
    filtered_signals = [signal for signal in signals if signal is not None]
    filtered_signals_list = list(filtered_signals)
    filtered_signals = [str(line).replace('\n', ' ') for line in filtered_signals]
    filtered_again = []
    for line in filtered_signals:
        if '16385' in line:
            filtered_again.append(line.replace('\n', ' '))
    
    print('This is the signal: ', filtered_again)
    
    import re

    open_positions = mt5.positions_get()
    pattern = r"(Buy|Sell) (\w+\.a)\s+-\s+(\w+\.a) at timeframe (\d+)\s+right now\s+Hedge Ratio: (\d+\.\d+)"
    counter = 0
    pairs = []

    for line in filtered_again:

        match = re.match(pattern, line)
        if match:
            side = match.group(1)
            pair_1 = match.group(2)
            pair_2 = match.group(3)
            tf = int(match.group(4))
            hr_ratio = float(match.group(5))
            pairs.append((side, pair_1, pair_2, tf, hr_ratio))


            lot = 0.15
            lot2 = float(round(lot * hr_ratio,2))
            symbol = pair_1 
            symbol2 = pair_2
            deviation = 5

            if side == 'Sell':

                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
                price2 = mt5.symbol_info_tick(symbol).bid
            elif side == 'Buy':

                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol2).ask
                price2 = mt5.symbol_info_tick(symbol2).ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": price,
                "deviation": deviation,
                "magic": 234000,
                "comment": f"{pair_1}@{tf}{side}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result1 = mt5.order_send(request)
            result1

            counter += 1
            print("Order {} for {}. {} lots at {} with deviation={} points at timeframe {}".format(counter, symbol, lot, price, deviation,tf))
            request2 = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol2,
                "volume": lot2,
                "type": order_type,
                "price": price2,
                "deviation": deviation,
                "magic": 234000,
                "comment": f"{pair_2}@{tf}{side}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result2 = mt5.order_send(request2)
            result2

            counter += 1
            print("Order {} for {}. {} lots at {} with deviation={} points at timeframe {}".format(counter, symbol2, lot2, price2, deviation,tf))

    ### Closing Logic

    import re
    positions = mt5.positions_get()
    pos_com = [i.comment for i in positions]
    comments = pos_com
    pattern = r"(\w+\.a)@(\d+)"
    timeframe_dict = {}

    for comment in comments:
        if comment.startswith("H"):
            continue
        match = re.match(pattern, comment)
        if match:
            symbol = match.group(1)
            timeframe = int(match.group(2))

            if timeframe == 16385:
                if timeframe not in timeframe_dict:
                    timeframe_dict[timeframe] = [symbol]
                else:
                    if symbol not in timeframe_dict[timeframe]:
                        timeframe_dict[timeframe].append(symbol)
        else:
            print('No matches for tf dict')

    def bb(df):
        sma = df.rolling(window=7).mean()
        std_dev = df.rolling(window=14).std()
        df['upper_band'] = sma + (2 * 2)
        df['lower_band'] = sma - (2 * 2)
        df['sma'] = sma
        return df['upper_band'], df['lower_band'], df['sma']

    def get_num_points(timeframe):
        bars_dict = {   
            1: 3600,
            2: 1800,
            3: 1000,
            4: 720,
            5: 1440,
            6: 1200,
            10: 720,
            12: 600,
            15: 960,
            30: 480,
            16385: 240
        }

        return bars_dict.get(timeframe)

    close_positions = []
    open_positions = mt5.positions_get()

    for timeframe, pairs in timeframe_dict.items():
        if timeframe == 16385:
            mt5_timeframe = mt5.TIMEFRAME_H1
        num_points = get_num_points(mt5_timeframe)

        print('this is how many pairs is on:', len(pairs))
        for i in range(0, len(pairs) -1, 2):
            symbol_1, symbol_2 = pairs[i], pairs[i+1]
            rates_1 = mt5.copy_rates_from_pos(symbol_1, mt5_timeframe, 0, num_points)
            rates_2 = mt5.copy_rates_from_pos(symbol_2, mt5_timeframe, 0, num_points)

            if len(open_positions) == 0:
                print('No Open Positions')
                break
            if rates_1 is not None and rates_2 is not None:
                df_1 = pd.DataFrame(rates_1)
                df_1['time'] = pd.to_datetime(df_1['time'], unit='s')
                df_2 = pd.DataFrame(rates_2)
                df_2['time'] = pd.to_datetime(df_2['time'], unit='s')

                merged_df = pd.merge(df_1, df_2, on='time', suffixes=(f'_{symbol_1}', f'_{symbol_2}'))
                merged_df = merged_df.T.drop_duplicates().T
                df1 = pd.DataFrame(merged_df[f'close_{symbol_1}'])
                df2 = pd.DataFrame(merged_df[f'close_{symbol_2}'])
                test2 = pd.DataFrame(df1[f'close_{symbol_1}'] - df2[f'close_{symbol_2}'])
                sma, upper_band, lower_band = bb(test2)
                # print(f'Running Closing Logic on {symbol_1} and {symbol_2} at {timeframe}min')

                for position in open_positions:
                    if f'{symbol_1}@{timeframe}' in position.comment or f'{symbol_2}@{timeframe}' in position.comment:
                        if 'Se' in position.comment:
                            pip_difference = round(10000 * (test2[0].iloc[-1] - test2.sma.iloc[-1]), 2)
                            if 'JPY' or 'jpy' in position.comment:
                                pip_difference = pip_difference / 100
                            print(f'{symbol_1} - {symbol_2} spread is currently', pip_difference, 'pips away from the SMA')
                            if test2[0].iloc[-1] < test2.sma.iloc[-1]:
                                close_positions.append(position)
                                print(f"{position.comment} to be closed")
                            else:
                                print(f'Keep {position.comment} open')

                        elif 'Bu' in position.comment:
                            pip_difference = round(10000 * (test2.sma.iloc[-1] - test2[0].iloc[-1]), 2)
                            if 'JPY' or 'jpy' in position.comment:
                                pip_difference = pip_difference / 100
                            print(f'{symbol_1} - {symbol_2} spread currently', pip_difference, 'pips away from the SMA')

                            if test2[0].iloc[-1] > test2.sma.iloc[-1]:
                                close_positions.append(position)
                                print(f"{position.comment} to be closed")
                            else:
                                print(f'Keep {position.comment} open')

    for position in close_positions:
        close_position(position)
        print(f'Closed {position.comment} at {position.profit}')

    ## Hedging Logic

    import re
    pattern = r"(Buy|Sell)\s+(\w+)\s+-\s+(\w+)\s+at\s+timeframe\s+(\d+)\s+right now\s+Hedge Ratio:\s+(\d+\.\d+)"
    counter = 0
    pairs = []

    for line in filtered_again:
        match = re.match(pattern, line)
        if match:
            pair_1 = match.group(2)
            pair_2 = match.group(3)
            pairs.append((pair_1, pair_2))
            
    over_exposed_syms = []
    OE = {}
    OE = get_exp(OE)
    if len(filtered_again) > 0:
        if len(OE.keys()) > 0:
            print('Previous Exposure:')
            OE = get_exp(OE)
            for key in OE.keys():
                for pair in pairs[0]:
                    if key in pair:
                        over_exposed_syms.append(key)
                        over_exposed_syms = list(set(over_exposed_syms))
                    hedge_order()
                    print(f'Hedge order for {key} sent')
                    break
            print('New Exposure:')
            OE = get_exp(OE)
        else:
            print('No Hedging required')
    import re
    positions = mt5.positions_get()
    pos_com = [i.comment for i in positions]
    comments = pos_com
    pattern = r"H(\w+(?:\.a)?)@(\d+)(Sell|Buy)" #Change 
    hedge_timeframe_dict = {}

    for comment in comments:
        match = re.match(pattern, comment)
        if match:
            symbol = match.group(1)
            timeframe = int(match.group(2))

            if timeframe == 16385: # CHANGE THIS TO CORRESPONDING SCRIPT
                if timeframe not in hedge_timeframe_dict:
                    hedge_timeframe_dict[timeframe] = [symbol]
                else:
                    if symbol not in hedge_timeframe_dict[timeframe]:
                        hedge_timeframe_dict[timeframe].append(symbol)
    print(f'Hedged Dictionary: {hedge_timeframe_dict}')

    timeframes = [mt5.TIMEFRAME_H1]

    bars_dict = {mt5.TIMEFRAME_H1: 240}
    def get_num_points(timeframe):
        bars_dict = {
            16385: 240,
        }

        return bars_dict.get(timeframe)
    
    if len(hedge_timeframe_dict) > 0:
        print('Running Hedge Close Logic')
        hedge_close_logic()
    else:
        print('No Hedged Positions Open')