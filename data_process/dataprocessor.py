import os
import time
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import logging
from my_parser import args


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='dataprocessor.log',
    filemode='w'
)

class StockDataProcessor:
    def __init__(self, price_dir, news_dir, news_dim):
        self.market_data = []
        self.news_data = []
        self.labels = []
        self.num_stocks = 0

        all_stock_files = sorted(os.listdir(price_dir))
        self.num_stocks = len(all_stock_files)


        stock_market_data = []
        stock_news_data = []
        stock_labels = []

        for stock_file in tqdm(all_stock_files):
            if not stock_file.endswith(".csv"):
                continue

            stock_id = stock_file.replace(".csv", "")
            price_path = os.path.join(price_dir, stock_file)
            price_data = pd.read_csv(price_path)

            if "Date" not in price_data.columns or "Label" not in price_data.columns:
                raise ValueError(f"{stock_file} has  no 'Date' pr 'Label' ")

            price_data = price_data.sort_values(by="Date").reset_index(drop=True)


            news_dict = {}
            news_folder = os.path.join(news_dir, stock_id)
            if os.path.exists(news_folder):
                for news_file in os.listdir(news_folder):
                    if news_file.endswith(".npy"):
                        news_date = news_file.replace(".npy", "")
                        news_path = os.path.join(news_folder, news_file)
                        news_dict[news_date] = np.load(news_path).astype(np.float32)


            market_windows = []
            news_windows = []
            labels = []

            for i in range(len(price_data)):
                date = price_data.iloc[i]["Date"]


                market_window = price_data.iloc[i].drop(["Date", "Label"]).values.astype(np.float32)
                # market_window = price_data.loc[i, ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values.astype(np.float32)


                news_window = np.zeros(news_dim, dtype=np.float32)  # 默认全 0
                if date in news_dict:
                    news_data = news_dict[date]

                    news_window = news_data.flatten()


                label = int(price_data.iloc[i]["Label"])

                # news_array = np.array(news_windows)
                # logging.debug(f"Stock {stock_file}: market_windows shape: {news_array.shape}")
                market_windows.append(market_window)
                news_windows.append(news_window)
                labels.append(label)


            stock_market_data.append(np.array(market_windows))
            stock_news_data.append(np.array(news_windows))
            stock_labels.append(np.array(labels))


        try:
            self.market_data = np.stack(stock_market_data, axis=1)
            self.news_data = np.stack(stock_news_data, axis=1)
            self.labels = np.stack(stock_labels, axis=1)
        except ValueError as e:
            logging.error(f"Error stacking arrays: {e}")
            for i, news_array in enumerate(stock_news_data):
                logging.error(f"Stock {all_stock_files[i]}: market_windows shape: {news_array.shape}")
            raise
    def save_to_pkl(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "market.pkl"), "wb") as f:
            pickle.dump(self.market_data, f)


        with open(os.path.join(output_dir, "news.pkl"), "wb") as f:
            pickle.dump(self.news_data, f)


        with open(os.path.join(output_dir, "labels.pkl"), "wb") as f:
            pickle.dump(self.labels, f)


class StockSentimentProcessor:
    def __init__(self, sentiment_dir, trading_date_path):
        self.sentiments = []

        all_stock_files = sorted(os.listdir(sentiment_dir))

        df = pd.read_csv(trading_date_path)

        df["Date"] = pd.to_datetime(df["Date"])


        trading_date = sorted(df["Date"].tolist())

        sentiment_list = []


        for date in tqdm(trading_date):
            stock_sentiments = []
            for stock_file in all_stock_files:
                stock_path = os.path.join(sentiment_dir, stock_file)
                date_path = os.path.join(stock_path, date.strftime("%Y-%m-%d") + ".npy")
                if os.path.exists(date_path):
                    sentiment_data = np.load(date_path).astype(np.float32)
                    stock_sentiments.append(sentiment_data)
                else:
                    print('no path')
                    break

            sentiment_list.append(np.array(stock_sentiments))

        self.sentiments = np.stack(sentiment_list, axis=0)

    def save_to_pkl(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)


        with open(os.path.join(output_dir, "sentiment.pkl"), "wb") as f:
            pickle.dump(self.sentiments, f)

if __name__ == "__main__":

    processor = StockDataProcessor(price_dir = f"../data/{args.dataset}/price/",
                                   news_dir = f"../data/{args.dataset}/news_embedding/",
                                   news_dim=20)
    processor.save_to_pkl(output_dir = f"..data/{args.dataset}/pkl/")
    print("input_pkl finished")

    processor = StockSentimentProcessor(sentiment_dir = f"../data/{args.dataset}/sentiment_for_relation/",
                                       trading_date_path = f"../data/{args.dataset}/trading_date_list.csv")
    processor.save_to_pkl(output_dir = f"../data/{args.dataset}/pkl/")
    print("sentiment_pkl finished")
