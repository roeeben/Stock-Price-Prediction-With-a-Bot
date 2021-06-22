from sklearn import preprocessing
import numpy as np
from finta import TA
import pandas as pd

def train_test_split_preparation(new_df, train_percentage, val_percentage, history_points):
    # Preparation of train test set.
    train_size = int(new_df.shape[0] * train_percentage)  # val size
    val_size = int(new_df.shape[0] * val_percentage)

    train_data = new_df[:train_size]
    val_data = new_df[train_size:train_size + val_size]
    test_data = new_df[train_size + val_size:]

    test_data = test_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    normalizer = preprocessing.MinMaxScaler()
    train_normalized_data = normalizer.fit_transform(train_data)

    test_normalized_data = normalizer.transform(test_data)
    val_normalized_data = normalizer.transform(val_data)

    X_train = np.array([train_normalized_data[:, 0:][i: i + history_points].copy() for i in
                        range(len(train_normalized_data) - history_points)])

    y_train = np.array([train_normalized_data[:, 0][i + history_points].copy() for i in
                        range(len(train_normalized_data) - history_points)])
    y_train = np.expand_dims(y_train, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    next_day_close_values = np.array(
        [train_data['Close'][i + history_points].copy() for i in range(len(train_data) - history_points)])
    next_day_close_values = np.expand_dims(next_day_close_values, -1)

    y_normaliser.fit(next_day_close_values)

    X_test = np.array([test_normalized_data[:, 0:][i: i + history_points].copy() for i in
                       range(len(test_normalized_data) - history_points)])

    y_test = np.array([test_data['Close'][i + history_points].copy() for i in range(len(test_data) - history_points)])

    y_test = np.expand_dims(y_test, -1)

    X_val = np.array([val_normalized_data[:, 0:][i: i + history_points].copy() for i in
                       range(len(val_normalized_data) - history_points)])

    y_val = np.array([val_data['Close'][i + history_points].copy() for i in range(len(val_data) - history_points)])

    y_val = np.expand_dims(y_val, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test, y_normaliser


def add_technical_indicators(new_df, stock_df):
    # Adding of technical indicators to data frame (Exponential moving average and Bollinger Band)
    edited_df = pd.DataFrame()

    edited_df['open'] = stock_df['Open']
    edited_df['high'] = stock_df['High']
    edited_df['low'] = stock_df['Low']
    edited_df['close'] = stock_df['Close']
    edited_df['volume'] = stock_df['Volume']
    edited_df.head()

    ema = TA.EMA(edited_df)
    bb = TA.BBANDS(edited_df)

    new_df['Exponential_moving_average'] = ema.copy()

    new_df = pd.concat([new_df, bb], axis=1)

    for i in range(19):
        new_df['BB_MIDDLE'][i] = new_df.loc[i, 'Exponential_moving_average']

        if i != 0:
            higher = new_df.loc[i, 'BB_MIDDLE'] + 2 * new_df['Close'].rolling(i + 1).std()[i]
            lower = new_df.loc[i, 'BB_MIDDLE'] - 2 * new_df['Close'].rolling(i + 1).std()[i]
            new_df['BB_UPPER'][i] = higher
            new_df['BB_LOWER'][i] = lower
        else:
            new_df['BB_UPPER'][i] = new_df.loc[i, 'BB_MIDDLE']
            new_df['BB_LOWER'][i] = new_df.loc[i, 'BB_MIDDLE']
    return new_df


def on_balance_volume_creation(stock_df):
    # Adding of on balance volume to dataframe

    new_df = pd.DataFrame({})

    new_df = stock_df[['Close']].copy()

    new_balance_volume = [0]
    tally = 0

    for i in range(1, len(new_df)):
        if (stock_df['Close'][i] > stock_df['Close'][i - 1]):
            tally += stock_df['Volume'][i]
        elif (stock_df['Close'][i] < stock_df['Close'][i - 1]):
            tally -= stock_df['Volume'][i]
        new_balance_volume.append(tally)

    new_df['On_Balance_Volume'] = new_balance_volume
    minimum = min(new_df['On_Balance_Volume'])

    new_df['On_Balance_Volume'] = new_df['On_Balance_Volume'] - minimum
    new_df['On_Balance_Volume'] = (new_df['On_Balance_Volume'] + 1).transform(np.log)

    return new_df

if __name__ == '__main__':
    train_split = 0.7

    history_points = 21
    X_train, y_train, X_test, y_test, y_reverse_normaliser = train_test_split_preparation(new_df2, train_split)