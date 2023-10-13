
def coint_finder2():
    print("Cointegration finder function!")


def coint_finder(symbol, timeframe, bar_count):
    
    def get_data(symbol, bar_count):
        '''Fetches price data for a symbol'''
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bar_count)
        return pd.DataFrame(rates)[['time', 'close']].set_index('time')

    def compute_spread(pair):
        '''Computes the spread for a given pair'''
        data1 = get_data(pair[0])
        data2 = get_data(pair[1])
        merged = data1.join(data2, lsuffix="_x", rsuffix="_y")
        spread = merged['close_x'] - merged['close_y']
        return spread.dropna()

    def adf_test(spread):
        '''Runs ADF test on a spread series'''
        result = adfuller(spread)
        return {'ADF Statistic': result[0], 'p-value': result[1], 'Critical Values': result[4]}
    
    # Running the tests
    results = {}
    for pair in first_order_pairs:
        print(f'Running through pair {pair}')
        spread = compute_spread(pair)
        results[pair] = adf_test(spread)

    # Convert results to a DataFrame
    df = pd.DataFrame(results).T
    
    coint_pairs = []

    for idx, row in df.iterrows():
        if row['ADF Statistic'] < row['Critical Values']['10%']:
            print(f'Pair {idx} is cointegrated')
            coint_pairs.append(idx)
            
    print(coint_pairs)
