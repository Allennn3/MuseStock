import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from my_parser import args
def count_null_and_na_values(price_path):
    count = 0
    null_columns = {}
    for ticker_csv in os.listdir(price_path):
        source_path = os.path.join(price_path, ticker_csv)
        data = pd.read_csv(source_path)

        # count null and na values
        null_count = data.isnull().sum().sum()
        na_count = data.isna().sum().sum()
        count += null_count + na_count

        # record column names
        if null_count > 0 or na_count > 0:

            null_columns[ticker_csv] = {
                'null_counts': {col: count for col, count in data.isnull().sum().to_dict().items() if count > 0},
                'na_counts': {col: count for col, count in data.isna().sum().to_dict().items() if count > 0}
            }

            print(f"Total null and na values in {ticker_csv}: {null_count + na_count}")
            print(f"Null values in columns: {null_columns[ticker_csv]['null_counts']}")
            print(f"Na values in columns: {null_columns[ticker_csv]['na_counts']}")

    print(f"Total null and na values across all files: {count}")
    print(f"Null and na values in each file: {null_columns}")



# linear interpolation
def linear_interpolation(data_dir, saved_dir):

    for ticker_csv in os.listdir(data_dir):
        source_path = os.path.join(data_dir, ticker_csv)
        saved_path = os.path.join(saved_dir, ticker_csv)

        data = pd.read_csv(source_path)

        # check data
        if data.isnull().values.any():
            # linear interpolation
            data.interpolate(method='linear', inplace=True)


            data.fillna(method='ffill', inplace=True)

            data.fillna(method='bfill', inplace=True)
            print(f"填充完成: {ticker_csv}")


        data.to_csv(saved_path, index=False)

def add_technical_indicators(data_dir, dataset_date):
    # compute SMA
    def calculate_SMA(df, period=20):
        return df['Close'].rolling(window=period, min_periods=1).mean()

    # compute EMA
    def calculate_EMA(df, period=12):
        return df['Close'].ewm(span=period, adjust=False).mean()

    # compute RSI
    def calculate_RSI(df, period=14):
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # compute MACD
    def calculate_MACD(df, fast_period=12, slow_period=26, signal_period=9):
        ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal

    # calculate Bollinger Bands
    def calculate_boll_ub(df, period=20, std_factor=2):
        sma = df['Close'].rolling(window=period, min_periods=1).mean()
        std = df['Close'].rolling(window=period, min_periods=1).std()
        upper_band = sma + std_factor * std
        return upper_band

    # calculate Bollinger Bands
    def calculate_boll_lb(df, period=20, std_factor=2):
        sma = df['Close'].rolling(window=period, min_periods=1).mean()
        std = df['Close'].rolling(window=period, min_periods=1).std()
        lower_band = sma - std_factor * std
        return lower_band

    # compute ATR
    def calculate_ATR(df, period=14):

        high = df['High']
        low = df['Low']
        close = df['Close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        return atr

    # compute CCI
    def calculate_CCI(df, period=20):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(window=period, min_periods=1).mean()
        mean_deviation = (typical_price - sma).abs().rolling(window=period, min_periods=1).mean()
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci

    # calculate Williams %R
    def calculate_WilliamsR(df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close']
        highest_high = high.rolling(window=period, min_periods=1).max()
        lowest_low = low.rolling(window=period, min_periods=1).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r


    for ticker_csv in os.listdir(data_dir):
        source_path = os.path.join(data_dir, ticker_csv)
        df = pd.read_csv(source_path)
        df['SMA_20'] = calculate_SMA(df, period=20)
        df['EMA_12'] = calculate_EMA(df, period=12)
        df['RSI_14'] = calculate_RSI(df, period=14)
        df['MACD'], df['MACD_Signal'] = calculate_MACD(df, fast_period=12, slow_period=26, signal_period=9)
        df['Boll_UB'] = calculate_boll_ub(df, period=20)
        df['Boll_LB'] = calculate_boll_lb(df, period=20)
        df['ATR_14'] = calculate_ATR(df, period=14)
        df['CCI_20'] = calculate_CCI(df, period=20)
        df['WilliamsR_14'] = calculate_WilliamsR(df, period=14)

        ACL_DATE = '2014-01-01'
        CMIN_US_DATE = '2018-01-02'
        CMIN_CN_DATE = '2018-01-01'
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Date'] > dataset_date]
        df.to_csv(source_path, index=False)
        print(f"{ticker_csv} technical indicators added.")
        # print(df.tail(1))

# generate label
def generate_label(price_path):

    for ticker_csv in os.listdir(price_path):
        if ticker_csv.endswith(".csv"):
            source_path = os.path.join(price_path, ticker_csv)
            df = pd.read_csv(source_path)
            df['Label'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df.at[df.index[-1], 'Label'] = 1  # 将最后一行的 Label 设为 1
            df.to_csv(source_path, index=False)

# standardization
def standardization(price_path):

    for ticker_csv in os.listdir(price_path):
        if ticker_csv.endswith(".csv"):
            print(ticker_csv)
            source_path = os.path.join(price_path, ticker_csv)
            df = pd.read_csv(source_path)

            # Min-Max
            minmax_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA_20', 'EMA_12',
                           'MACD', 'MACD_Signal', 'Boll_UB', 'Boll_LB', 'CCI_20', 'ATR_14', 'RSI_14', 'WilliamsR_14']
            scaler_minmax = MinMaxScaler(feature_range=(0, 1))
            df[minmax_cols] = scaler_minmax.fit_transform(df[minmax_cols])

            df.to_csv(source_path, index=False)

if __name__ == '__main__':
    # compute null and na values
    count_null_and_na_values(price_path=f"../data/{args.dataset}/price/")
    # fill null and na values
    linear_interpolation(data_dir = f"../data/{args.dataset}/price/",
                        saved_dir = f"../data/{args.dataset}/price/")
    # add technical indicators
    add_technical_indicators(data_dir = f"../data/{args.dataset}/price/", dataset_date="2018-01-01")
    # standardization
    standardization(price_path = f"../data/{args.dataset}/price/")
    # generate label
    generate_label(price_path = f"../data/{args.dataset}/price/")




