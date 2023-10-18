# Libraries and Modules used
import MetaTrader5 as mt5 
import pandas as pd
from sklearn.linear_model import LinearRegression
import ta
import warnings
from statsmodels.tsa.stattools import adfuller 
warnings.filterwarnings("ignore")
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import scipy.optimize as opt
import pandas_ta as ta
import datetime 
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis

mt5.initialize()
# Replace following with your MT5 Account Login
account=51127988 # 
password="Aar2frM7"
server = 'ICMarkets-Demo'

def run():
    first_order_pairs = [('AUDUSD.a', 'NZDUSD.a'), 
                        ('EURUSD.a', 'GBPUSD.a'),
                        ('EURNZD.a', 'GBPNZD.a')]
    # Functions # 
    def get_rates(pair1, timeframe, x):
        pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, timeframe, 0, x))
        pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')
        return pair1['close']
    
    def get_data(symbol, bars=6000):
        rates = pd.DataFrame(mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, bars))
        rates['time'] = pd.to_datetime(rates['time'], unit = 's')
        return rates[['time', 'close']].set_index('time')

    def compute_spread(pair):
        data1 = get_data(pair[0])
        data2 = get_data(pair[1])
        merged = data1.join(data2, lsuffix="_x", rsuffix="_y")
        spread = merged['close_x'] - merged['close_y']
        return spread.dropna()

    def adf_test(spread):
        '''Runs ADF test on a spread series'''
        result = adfuller(spread)
        return {'ADF Statistic': result[0], 'p-value': result[1], 'Critical Values': result[4]}
    
    def rsi(data, length):

        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
    
    def generate_features(data):
        fiften_day_avg = data.rolling(window=15).mean().round(5)
        sixty_day_avg = data.rolling(window=60).mean().round(5)

        features_df = pd.DataFrame(index=data.index)
        n = 10
        features_df['close'] = data
        features_df['Shifted_Close'] = data.shift(1)
        features_df['momentum'] = data - data.shift(n)
        features_df['fiften_day_avg'] = fiften_day_avg
        features_df['sixty_day_avg'] = sixty_day_avg
        features_df['RSI'] = rsi(data, length=14)
        features_df['ROC'] = ((data - data.shift(n)) / data.shift(n)) * 100
        
        # Remove rows with any NA values
        features_df.dropna(inplace=True)
        return features_df

    def train_neural_network(dataframe):
        n_features = dataframe.shape[1]
        X_train = dataframe.drop(columns='close')
        y_train = dataframe['close']

        # 1. Train a simple neural network
        model = Sequential()
        model.add(Input(shape=(n_features-1,))) # Excluding 'close' column
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))  # We'll extract activations from this layer
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=100, batch_size=32)

        # 2. Extract activations from the second last layer
        activation_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        activations = activation_model.predict(X_train)

        # 3. Cluster on these activations
        activations_normalized = StandardScaler().fit_transform(activations)
        db = DBSCAN(eps=0.5, min_samples=5).fit(activations_normalized)
        
        labels = db.labels_
        
        return labels, activation_model, db 

    # Define the objective function to minimize (MSE)
    def objective(params, X_test, y_test):
        predicted = np.dot(X_test, params)
        mse = np.mean((predicted - y_test) ** 2)
        return mse

    def substate_statistics(df):
    
        statistics_list = []

        grouped = df.groupby('sub_state')['pct_change']
        # Calculate statistics for each group
        for sub_state, pct_changes in grouped:
            mean_values = pct_changes.mean()
            mode_values = pct_changes.mode()[0] if not pct_changes.mode().empty else np.nan
            std_values = pct_changes.std()
            var_values = pct_changes.var()
            kurt_values = kurtosis(pct_changes, fisher=True)

            fifteen_MA = pct_changes.rolling(window=15).mean()
            forty_five_MA = pct_changes.rolling(window=45).mean()
            ratio = fifteen_MA / forty_five_MA

            q1 = ratio.quantile(0.25)
            q3 = ratio.quantile(0.75)
            iqr = q3 - q1

            # Append statistics to the list
            statistics_list.append({
                'sub_state': sub_state,
                'Mean': mean_values,
                'Mode': mode_values,
                'STD': std_values,
                'VAR': var_values,
                'Kurtosis': kurt_values,
                'Q1': q1,
                'Q3': q3,
                'IQR': iqr
            })
            
        statistics_df = pd.DataFrame(statistics_list)
        
        statistics_df['Q1'].fillna(0, inplace=True)
        statistics_df['Q3'].fillna(0, inplace=True)
        statistics_df['IQR'].fillna(0, inplace=True)

        # Now, split the data
        df['Label'] = df['Price'].diff().apply(lambda x: 'up' if x > 0 else 'down')
        X = df[['pct_change', 'RSI']].dropna()
        y = df['Label'].dropna()

        sub_states = df['sub_state']
        # print('substates are ', sub_states)
        X_train, X_test, y_train, y_test, sub_states_train, sub_states_test = train_test_split(
            X, y, sub_states, test_size=0.2, random_state=42
        )
        # Step 1: Fit a Random Forest model to determine feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Get feature importances
        importances = rf.feature_importances_

        # Step 2: Select important features based on feature importances
        # (Here, we're arbitrarily choosing to keep the top 10 features)
        important_feature_indices = np.argsort(importances)[::-1][:10]

        # Extract the important features from the original data
        important_feature_names = X_train.columns[important_feature_indices]
        X_train_important = X_train[important_feature_names]
        X_test_important = X_test[important_feature_names]

        # Step 3: Train an SVM model using only important features
        svm = SVC(kernel='linear', C=1, probability=True)  # Note the `probability=True`
        svm.fit(X_train_important, y_train)

        # Evaluate the model (Optional)
        score = svm.score(X_test_important, y_test)
        print(f"SVM model accuracy: {score * 100:.2f}%")

        # Step 4: Get confidence scores (continuous values between 0 and 1)
        confidence_scores = svm.predict_proba(X_test_important)[:, 1]

        # Normalize the confidence_scores to be between 0 and 1
        scaler = MinMaxScaler((0, 1))
        confidence_scores = scaler.fit_transform(confidence_scores.reshape(-1, 1))
        confidence_scores = confidence_scores.flatten()
        confidence_scores_df = pd.DataFrame({'Confidence_Score': confidence_scores, 'sub_state': sub_states_test.reset_index(drop=True)})
        aggregated_scores = confidence_scores_df.groupby('sub_state')['Confidence_Score'].mean()
        # print('aggregated scores are ', aggregated_scores)
        # Convert to DataFrame for better visualization
        confidence_df = pd.DataFrame({'Confidence_Score': confidence_scores.flatten()})
        print(confidence_df.head())

        statistics_df['SVM_Score'] = statistics_df['sub_state'].map(aggregated_scores)
        statistics_df['SVM_Score'].fillna(0.5, inplace=True)
        missing_sub_states = set(statistics_df['sub_state']) - set(aggregated_scores.index)

        statistics_df['Score'] = statistics_df.apply(calculate_score, axis=1)
        statistics_df['Q1'].fillna(0, inplace=True)
        statistics_df['Q3'].fillna(0, inplace=True)
        statistics_df['IQR'].fillna(0, inplace=True)

        return statistics_df[['sub_state', 'Score']]

    def calculate_score(row):
        score = 0
        # Weights for each statistic
        weights = {
            'Mean': 2,
            'Mode': 1,
            'STD': -1,
            'VAR': -1,
            'Kurtosis': -1,
            'Q1': 1,
            'Q3': 1,
            'IQR': -1,
            'SVM_Score': 2,
        }
        
        if row['SVM_Score'] > 0.500001:
            score += weights['SVM_Score'] * row['SVM_Score']
        elif row['SVM_Score'] < 0.499999:
            score -= weights['SVM_Score'] * row['SVM_Score']
            
        if row['Mean'] < 0 and row['Mode'] < 0:
            weights['Mean'] = 2.75
            weights['Mode'] = 1.5
        elif row['Mean'] > 0 and row['Mode'] > 0:
            weights['Mean'] = 2.75
            weights['Mode'] = 1.5

        score += weights['Mean'] * row['Mean']
        score += weights['Mode'] * row['Mode']
        score += weights['STD'] * row['STD']
        score += weights['VAR'] * row['VAR']
        score += weights['Kurtosis'] * np.log1p(abs(row['Kurtosis']))
        
        if row['Q1'] > 0:
            score += weights['Q1'] * row['Q1']
            score += weights['Q3'] * row['Q3']
            score += weights['IQR'] * row['IQR']

        return score
    
    def calc_score(df, is_substate = False):
        statistics_list = []
        state_scores = {}
        
        # Loop through each unique state
        for unique_state in df['State'].unique():
            
            # Filter data for the current state
            state_df = df[df['State'] == unique_state]
            pct_changes = state_df['pct_change'].dropna()
            
            # Calculate statistics
            mean_values = pct_changes.mean()
            mode_values = pct_changes.mode()[0] if not pct_changes.mode().empty else np.nan
            std_values = pct_changes.std()
            var_values = pct_changes.var()
            kurt_values = kurtosis(pct_changes, fisher=True)
            
            fifteen_MA = pct_changes.rolling(window=15).mean()
            forty_five_MA = pct_changes.rolling(window=45).mean()
            ratio = (fifteen_MA / forty_five_MA).dropna()
            
            q1 = ratio.quantile(0.25)
            q3 = ratio.quantile(0.75)
            iqr = q3 - q1
            
            # Append statistics to the list
            statistics_list.append({
                'state': unique_state,
                'Mean': mean_values,
                'Mode': mode_values,
                'STD': std_values,
                'VAR': var_values,
                'Kurtosis': kurt_values,
                'Q1': q1,
                'Q3': q3,
                'IQR': iqr
            })
            
        # Convert the list of dictionaries to a DataFrame
        statistics_df = pd.DataFrame(statistics_list)
        
        # Now, split the data
        df['Label'] = df['Price'].diff().apply(lambda x: 'up' if x > 0 else 'down')
        X = df[['pct_change']].dropna()
        y = df['Label'].dropna()
        y = y.iloc[1:]
    
        states = df['State'].iloc[1:]
        
        # print(f"states are {states}")
        # print('substates are ', sub_states)
        X_train, X_test, y_train, y_test, states_train, states_test = train_test_split(
            X, y, states, test_size=0.2, random_state=42
        )
        # Step 1: Fit a Random Forest model to determine feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Get feature importances
        importances = rf.feature_importances_

        # Step 2: Select important features based on feature importances
        # (Here, we're arbitrarily choosing to keep the top 10 features)
        important_feature_indices = np.argsort(importances)[::-1][:10]

        # Extract the important features from the original data
        important_feature_names = X_train.columns[important_feature_indices]
        X_train_important = X_train[important_feature_names]
        X_test_important = X_test[important_feature_names]

        # Step 3: Train an SVM model using only important features
        svm = SVC(kernel='linear', C=1, probability=True)  # Note the `probability=True`
        svm.fit(X_train_important, y_train)

        # Evaluate the model (Optional)
        score = svm.score(X_test_important, y_test)
        print(f"SVM model accuracy: {score * 100:.2f}%")

        # Step 4: Get confidence scores (continuous values between 0 and 1)
        confidence_scores = svm.predict_proba(X_test_important)[:, 1]

        # Normalize the confidence_scores to be between 0 and 1
        scaler = MinMaxScaler((0, 1))
        confidence_scores = scaler.fit_transform(confidence_scores.reshape(-1, 1))
        confidence_scores = confidence_scores.flatten()
        confidence_scores_df = pd.DataFrame({'Confidence_Score': confidence_scores, 'state': states_test.reset_index(drop=True)})
        aggregated_scores = confidence_scores_df.groupby('state')['Confidence_Score'].mean()
        # Convert to DataFrame for better visualization
        confidence_df = pd.DataFrame({'Confidence_Score': confidence_scores.flatten()})
        # print(confidence_df.head())
        # print('stats state', statistics_df['state'])
        
        # missing_sub_states = set(statistics_df['state']) - set(aggregated_scores.index)
        # statistics_df['Score'] = statistics_df.apply(calculate_score, axis=1)
        statistics_df['Q1'].fillna(0, inplace=True)
        statistics_df['Q3'].fillna(0, inplace=True)
        statistics_df['IQR'].fillna(0, inplace=True)
            
        if is_substate:
            state_label = 'sub_state'
            statistics_df['SVM_Score'] = statistics_df['sub_state'].map(aggregated_scores)
        else:
            state_label = 'state'
            statistics_df['SVM_Score'] = statistics_df['state'].map(aggregated_scores)
        
        aggregated_scores = aggregated_scores.dropna()  # Drop NaN from aggregated_scores
        statistics_df['SVM_Score'] = statistics_df['state'].map(aggregated_scores)
        statistics_df['SVM_Score'].fillna(0.5, inplace=True)
        statistics_df['Score'] = statistics_df.apply(calculate_score, axis=1)
        
        for index, row in statistics_df.iterrows():
            state_scores[row['state']] = row['Score']
            
        return state_scores

    def is_order_open(pair, order_type, model):
        
        open_positions = mt5.positions_get()
        positions = []

        if model == 'lin':
            
            for i in open_positions:
                if 'Regress' in i.comment:
                    positions.append(i)

        elif model == 'ARIMA':
            
            for i in open_positions:
                if 'ARIMA' in i.comment:
                    positions.append(i)
                    
        elif model == 'MC_REGRESS':

            for i in open_positions:
                if 'MC_REGRESS' in i.comment:
                    positions.append(i) 

        for position in positions:
            # Define the position type
            pos_type = 'buy' if position.type == 0 else 'sell'
            
            # Check the symbol and the type
            if position.symbol == pair and pos_type == order_type:
                return True
        # If loop finishes and no matching open order was found
        return False

    def get_rates(pair1, tf, x):
        pair1 = pd.DataFrame(mt5.copy_rates_from_pos(pair1, tf, 0, x))
        pair1['time'] = pd.to_datetime(pair1['time'], unit = 's')
        return pair1['close']

    def calc_hedge_ratio(x, y):
        Model2 = sm.OLS(x, y)
        Hedge_Ratio2 = Model2.fit()
        hedge_ratio_float2 = round(Hedge_Ratio2.params[0], 2)
        return hedge_ratio_float2
    
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

    def lin_send_order():

        # Then continue with your order opening logic
        lot = 0.5
        for pair in orders['sell']:
            if not is_order_open(pair, 'sell', 'lin'):
                send_order(pair, 'sell', lot, 'S_Regress')

        for pair in orders['buy']:
            if not is_order_open(pair, 'buy', 'lin'):
                for key, val in hedge_ratios.items():
                    if pair in key:
                        send_order(pair, 'buy', round(float(lot * val), 2), 'B_Regress')
                    else: 
                        continue

    def MC_send_order(symbol, side, lot, comment, final_direction):
        
        lot = abs(round(float(lot * score), 2))
        print(f"for {symbol}, the lot size is {lot}")
        
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
    
    def mc_ordersending(final_direction):

        hedge_ratios = {}

        for i in coint_pairs:
            x = get_rates(i[0], mt5.TIMEFRAME_H4, 6000)
            y = get_rates(i[1], mt5.TIMEFRAME_H4, 6000)
            hedge_ratios[i] = calc_hedge_ratio(x, y)

        print(hedge_ratios)

        lot = 2.00

        # For selling orders
        for i in MC_orders['sell']:
            # Check if sell order is not already opened
            if not is_order_open(i, 'sell', 'MC_REGRESS'):
                MC_send_order(i, 'sell', lot, 'MC_REGRESS', score)

        # For buying orders
        for i in MC_orders['buy']:
            for key, val in hedge_ratios.items():
                if i == key[1]:  # We apply hedge ratio to the second pair
                    adjusted_lot = lot * val
                    # Check if buy order is not already opened
                    if not is_order_open(i, 'buy', 'MC_REGRESS'):
                        MC_send_order(i, 'buy', adjusted_lot, 'MC_REGRESS', final_direction)

    def multi_lin_ordersend():
    # For selling orders
        for i in orders['sell']:
            # Check if sell order is not already opened
            if not is_order_open(i, 'sell', 'lin'):
                print(f'Selling {i} (Regresion Model)')
                send_order(i, 'sell', lot, 'S_Regress')

        # For buying orders
        for i in orders['buy']:
            for key, val in hedge_ratios.items():
                # print(f'Looping through {i} and {key[1]}')
                if i == key[1]:  # We apply hedge ratio to the second pair

                    adjusted_lot = lot * val
                    print(adjusted_lot)
                    # Check if buy order is not already opened
                    if not is_order_open(i, 'buy', 'lin'):
                        send_order(i, 'buy', adjusted_lot, 'B_Regress')
                elif i == key[0]:
                    adjusted_lot = round((lot / val), 2)
                    if not is_order_open(i, 'buy', 'lin'):
                        print(f'Buying {i} (Regression Model)')
                        send_order(i, 'buy', adjusted_lot, 'B_Regress')
                    
    # Classes # 
    class MarkovChain:
        def __init__(self, states, states_dict):
            self.states = states
            self.transition_matrix = {}
            self.classifier = StateClassifier(states_dict)
            self.previous_state = None
            self.states_dict = states_dict
            
        def current_state(self, pair, current_sample):
            activation_model = self.states_dict[pair]['activation_model']
            current_activations = activation_model.predict(current_sample)

            # Use DBSCAN to identify the state
            db = self.states_dict[pair]['DB_scan']
            closest_index, _ = pairwise_distances_argmin_min(current_activations, db.components_)
            current_state = db.labels_[closest_index][0]

            return current_state
        
        def update_transition_matrix(self, current_state, pair):
            # Check and initialize the outer dictionary for the pair if needed
            if pair not in self.transition_matrix:
                self.transition_matrix[pair] = {}

            # Check and initialize the second level dictionary for previous_state if needed
            if self.previous_state not in self.transition_matrix[pair]:
                self.transition_matrix[pair][self.previous_state] = {}

            # Check and initialize the innermost dictionary for current_state if needed
            if current_state not in self.transition_matrix[pair][self.previous_state]:
                self.transition_matrix[pair][self.previous_state][current_state] = 0

            # Now you can safely update the count
            self.transition_matrix[pair][self.previous_state][current_state] += 1

        def classify_samples(self, samples, pair):
            return [self.classifier.classify_sample(sample.reshape(1, -1), pair) for sample in samples]

        def get_transition_matrix(self):
            return self.transition_matrix
        
        def create_transition_matrix(self, transitions_dict):

            result = {}

            for pair, transitions in transitions_dict.items():
                transition_counts = {}
                for from_state, to_states in transitions.items():
                    for to_state, count in to_states.items():
                        if from_state not in transition_counts:
                            transition_counts[from_state] = {}
                        if to_state not in transition_counts[from_state]:
                            transition_counts[from_state][to_state] = 0
                        transition_counts[from_state][to_state] += count

                probability_matrix = {}
                for from_state, to_states in transition_counts.items():
                    total_transitions = sum(to_states.values())
                    probability_matrix[from_state] = {to_state: count / total_transitions for to_state, count in to_states.items()}

                result[pair] = probability_matrix

            return result
        
        def substate_update_transition_matrix(self, original_matrix, pair, sub_states, meta_state, next_meta_state=None):
            new_transitions = defaultdict(lambda: defaultdict(int))
            sub_states = [int(s) for s in sub_states]
            meta_state = int(meta_state)  # Ensure meta_state is a native Python integer

            if meta_state not in new_transitions:
                new_transitions[meta_state] = defaultdict(int)
                # print('metastate not in new_transitions. adding now.')
            if f"{meta_state}-{sub_states[0]}" not in new_transitions[meta_state]:
                # print('metastate not in new_transitions meta state. adding now')
                new_transitions[meta_state][f"{meta_state}-{sub_states[0]}"] = 0
                
            new_transitions[meta_state][f"{meta_state}-{sub_states[0]}"] += 1
            print(f'Added count of {new_transitions[meta_state]}')
            
            # Adding transitions between sub-states
            for i in range(len(sub_states) - 1):
                from_state = f"{meta_state}-{sub_states[i]}"
                to_state = f"{meta_state}-{sub_states[i + 1]}"
                new_transitions[from_state][to_state] += 1

            # Adding transitions from meta-state to the first sub-state in each sequence
            new_transitions[meta_state][f"{meta_state}-{sub_states[0]}"] += 1

            # Adding transitions from the last sub-state in each sequence to the next meta-state if provided
            if next_meta_state is not None:
                new_transitions[f"{meta_state}-{sub_states[-1]}"][next_meta_state] += 1
            else:
                # If next meta-state is not provided, transition back to the same meta-state
                new_transitions[f"{meta_state}-{sub_states[-1]}"][meta_state] += 1

            # Merge new transitions into the original matrix for the specific pair
            if pair not in original_matrix:
                original_matrix[pair] = {}

            # Merge new transitions into the original matrix for the specific pair
            if pair not in original_matrix:
                original_matrix[pair] = {}

            for from_state, to_states in new_transitions.items():
                if from_state not in original_matrix[pair]:
                    # print(f"Adding new from_state {from_state} to original_matrix")
                    original_matrix[pair][from_state] = {}

                for to_state, count in to_states.items():
                    if to_state not in original_matrix[pair][from_state]:
                        # print(f"Adding new to_state {to_state} to original_matrix[{pair}][{from_state}]")
                        original_matrix[pair][from_state][to_state] = 0

                    original_matrix[pair][from_state][to_state] += count

        # Function to create a new probability matrix
        def substate_create_new_probability_matrix(self, original_matrix):
            new_prob_matrix = {}
            for pair, transitions in original_matrix.items():
                pair_prob_matrix = {}
                for from_state, to_states in transitions.items():
                    # Remove this line to include all types of states
                    if isinstance(from_state, (int, np.int64)) or (isinstance(from_state, str)):  # Add this line to check the type of key
                        total_transitions = sum(to_states.values())
                        pair_prob_matrix[from_state] = {to_state: count / total_transitions for to_state, count in to_states.items()}
                new_prob_matrix[pair] = pair_prob_matrix
            return new_prob_matrix

    class StateClassifier:
        def __init__(self, states_dict):
            self.states_dict = states_dict
            # Initialize NearestNeighbors models for each pair
            self.nn_models = {}
            for pair, values in self.states_dict.items():
                activations = values["activation_model"].predict(values["data"])  # Assuming "data" contains original features for each pair
                self.nn_models[pair] = NearestNeighbors(n_neighbors=1).fit(activations)

        def classify_sample(self, sample, pair):
            activation = self.states_dict[pair]["activation_model"].predict(sample)
            distance, index = self.nn_models[pair].kneighbors(activation)
            states = self.states_dict[pair]["states"]
            state = states[index[0][0]]
            return state

    print('Performing Loop')

    # Find Cointegrating Pairs # 

    results = {}

    for pair in first_order_pairs:
        print(f'Running through pair {pair}')
        spread = compute_spread(pair)
        results[pair] = adf_test(spread)

    df = pd.DataFrame(results).T
        
    coint_pairs = []

    for idx, row in df.iterrows():
        if row['ADF Statistic'] < row['Critical Values']['10%']:
            print(f'Pair {idx} is cointegrated')
            coint_pairs.append(idx)
            
    coint_dict = {}

    for pair in coint_pairs:
        coint_dict[pair] = compute_spread(pair)

    
    # Create Features # 
    head_features = {}

    for pair, data in coint_dict.items():
        features = generate_features(data)
        head_features[pair] = features
    
    # Find States with a Neural Network # 

    states_dict = {}

    for pair, dataframe in head_features.items():
        states, activation_model, db = train_neural_network(dataframe)
        states_dict[pair] = {
            "states": states,
            "activation_model": activation_model,
            "DB_scan": db,
            "data": dataframe.drop(columns='close')  # Store data for NearestNeighbors fitting in StateClassifier
        }

    orders = {
    "buy": [],
    "sell": []
    } 

    print('Performing Regression Model')

    from sklearn.neighbors import NearestNeighbors
    nearest_neighbor_models = {}  # A dictionary to hold the trained NearestNeighbors models for each pair.

    predictions_today = {} 
    predictions_tomorrow = {} 
    change_in_predictions = {}

    current_datetime = datetime.datetime.now()
    current_date_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')

    lin_markov_chain = MarkovChain(states, states_dict)

    print('Beginning Optimized Standard Linear Regression')
    print('')
    # for pair, features_df in features_dict.items():
    for pair, features_df in head_features.items(): 
        # Prepare the data
        X = features_df.drop(columns = ['close']).values[:-2]  # Exclude last two values for today's and tomorrow's prediction
        y = features_df['close'].values[:-2]  # Similarly, exclude the last two values 

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        sample = features_df.drop(columns=['close']).iloc[-1].values.reshape(1,-1)
        current_state = lin_markov_chain.classifier.classify_sample(sample, pair)
                                                                    
        # Classify samples in X_train and X_test into states
        states_train = lin_markov_chain.classify_samples(X_train, pair)
        
        # Update transition matrix based on the states sequence
        for current_state in states_train:
            if lin_markov_chain.previous_state is not None:
                lin_markov_chain.update_transition_matrix(current_state, pair)
            lin_markov_chain.previous_state = current_state
            
        # Optimization with basinhopping
        initial_params = np.ones(X_test.shape[1])
        result = opt.basinhopping(objective, initial_params, niter=100, stepsize=0.5, minimizer_kwargs={'args': (X_test, y_test)})
        optimized_params = result.x
        # Using the optimized parameters to make predictions
        prediction_today = np.dot(features_df.drop(columns = ['close']).iloc[-2, :].values, optimized_params)
        prediction_tomorrow = np.dot(features_df.drop(columns = ['close']).iloc[-1, :].values, optimized_params)

        predictions_today[pair] = prediction_today
        predictions_tomorrow[pair] = prediction_tomorrow
        change_in_predictions[pair] = prediction_today - np.dot(features_df.drop(columns = ['close']).iloc[-3, :].values, optimized_params)

    print(f"Time is {current_date_str}.")
    print('')
    
    for pair in predictions_today.keys():
        first_pair, second_pair = pair  # split the pair into individual currencies
        
        current_price = round(compute_spread(pair).iloc[-1], 5)
        print(f"For {pair}:")
        print(f"Today's prediction: {predictions_today[pair]:.5f}. Current price: {current_price}")
        
        if predictions_today[pair] > current_price:  # predicted spread is widening
            print(f"Signal: Sell {first_pair}, Buy {second_pair}")
            orders["sell"].append(first_pair)
            orders["buy"].append(second_pair)
        elif predictions_today[pair] < current_price:  # predicted spread is contracting
            print(f"Signal: Buy {first_pair}, Sell {second_pair}")
            orders["buy"].append(first_pair)
            orders["sell"].append(second_pair)
        print("-----")

    ## Matrix Creation ##
    lin_transition_matrix = lin_markov_chain.get_transition_matrix()
    transition_matrix_final = lin_markov_chain.create_transition_matrix(lin_transition_matrix)

    # Getting Most Common States to find Sub-States within Matrix
    most_common_states = {}

    # Loop through each pair and its transitions in lin_trans_matrix
    for pair, transitions in lin_transition_matrix.items():
        state_counts = defaultdict(int)
        # Check if we're dealing with sub-states
        if pair in transitions:
            # Handle sub-states separately
            sub_states = transitions[pair]
            for inner_states in sub_states.values():
                for state_key, count in inner_states.items():
                    state_counts[state_key] += count
        else:
            # Handle regular states
            for inner_states in transitions.values():
                for state_key, count in inner_states.items():
                    state_counts[state_key] += count

        # Find the most common state for the current dictionary and store it
        most_common_state = max(state_counts, key=state_counts.get)
        most_common_states[pair] = most_common_state
    
    # Initialize a dictionary to store DataFrames of most common states
    most_common_states_dfs = {}

    # Loop through each pair and its most common state in most_common_states
    for pair, most_common_state in most_common_states.items():
        # Create a DataFrame for the current pair
        pair_df = pd.DataFrame(
            head_features[pair]['close'].values, 
            states_dict[pair]['states']
        ).reset_index()
        
        pair_df.columns = ['State', 'Price']
        pair_df['pct_change'] = pair_df['Price'].pct_change()
        pair_df['RSI'] = ta.rsi(pair_df['Price'], length=14)
        
        # Filter rows in pair_df that match the most common state
        filtered_df = pair_df[pair_df['State'] == most_common_state]
        
        # Store the filtered DataFrame in most_common_states_dfs
        most_common_states_dfs[pair] = filtered_df

    all_substate_scores = {}
    scaler = StandardScaler()

    print("Beginning Order Sending")

    # Get the new probability matrix for all normal states + substates of the meta state #
    for pair, df in most_common_states_dfs.items():
        df = df.dropna()
        scaled_data = scaler.fit_transform(df[['Price', 'pct_change', 'RSI']])
        n_components = 3  # Number of sub-states
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)
        model.fit(scaled_data)
        sub_states = model.predict(scaled_data)
        
        df['sub_state'] = sub_states
        substate_scores = substate_statistics(df)
        all_substate_scores[pair] = substate_scores
        
        # Update original transition matrix
        meta_state = int(most_common_states[pair])  # The most common state for this pair

        lin_markov_chain.substate_update_transition_matrix(transition_matrix_final, pair, sub_states, meta_state)

        # Create a new probability matrix
        new_prob_matrix = lin_markov_chain.substate_create_new_probability_matrix(transition_matrix_final)

    
        all_pair_scores = {}

        for pair, most_common_state in most_common_states.items():
            print(f'Iterating through {pair}')
            # Create a DataFrame for the current pair
            pair_df = pd.DataFrame(
                head_features[pair]['close'].values, 
                states_dict[pair]['states']
            ).reset_index()
            
            pair_df.columns = ['State', 'Price']
            pair_df['pct_change'] = pair_df['Price'].pct_change()
            # print(pair_df)
            all_pair_scores[pair] = calc_score(pair_df)

    MC_orders = {
    'sell': [],
    'buy': []
    }

    print('MC_ORDERS')
    print(MC_orders)

    for pair, states_prob in new_prob_matrix.items():
        print(f"Looping through {pair}")
        current_state = states_dict[pair]['states'][-1]
    
        state_prob = states_prob.get(current_state, {})
        
        # Get the scores for this pair from the all_pair_scores and all_substate_scores dictionaries
        pair_scores = all_pair_scores.get(pair, {})
        substate_scores = all_substate_scores.get(pair, {})
        final_direction = 0
        
        for state, prob in state_prob.items():
            # Check if the state is a sub-state
            if isinstance(state, str) and '-' in state:
                
                # print(f' {state[2]} is a substate')
                substate = int(state.split('-')[1])  # Extract the substate index

                # Get the score for this sub-state from the substate_scores dictionary
                substate_score = all_substate_scores[pair].loc[substate]['Score']  # Default to 0 if the sub-state is not found
                
                # Calculate the weighted score for this sub-state
                weighted_substate_score = substate_score * prob
                
                # Update the final direction
                final_direction += weighted_substate_score
                
            else:
                # Get the score for this state from the pair_scores dictionary
                score = pair_scores.get(state, 0)  # Default to 0 if the state is not found

                # Calculate the weighted score for this state
                weighted_score = score * prob
                
                # Update the final direction
                final_direction += weighted_score

            print(f'Direction for {pair} is ', final_direction)
            
            if final_direction < 0:
                print(f"Selling {pair[0]} and buying {pair[1]}")
                MC_orders['sell'].append(pair[0])
                MC_orders['buy'].append(pair[1])
            elif final_direction > 0:
                print(f"Buying {pair[0]} and selling {pair[1]}")
                MC_orders['buy'].append(pair[0])
                MC_orders['sell'].append(pair[1])

        # Markov Chain Multiple Linear Regression Order # 
    print("Sending MC_Regress Orders now")
    mc_ordersending(final_direction)


    # Normal Multiple Linear Regression Order #
    hedge_ratios = {}

    for i in coint_pairs:
        x = get_rates(i[0], mt5.TIMEFRAME_H4, 6000)
        y = get_rates(i[1], mt5.TIMEFRAME_H4, 6000)
        hedge_ratios[i] = calc_hedge_ratio(x, y)

    lot = 0.75
    multi_lin_ordersend()

if __name__ == "__main__":
    run()