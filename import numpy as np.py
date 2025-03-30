import numpy as np
import pandas as pd
import yfinance as yf
from openpyxl import Workbook

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

def calculate_sma(df, period=50):
    df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    return df

def calculate_ema(df, period=50):
    df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = sma + (std_dev * std)
    df['BB_Lower'] = sma - (std_dev * std)
    return df

def calculate_adx(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (-minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(span=period, adjust=False).mean()
    
    return df

def calculate_supertrend(df, period=10, multiplier=4):
    hl2 = (df['High'] + df['Low']) / 2
    atr = df['High'].rolling(period).max() - df['Low'].rolling(period).min()
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    
    df['Supertrend'] = np.where(df['Close'] > basic_upperband, basic_upperband, basic_lowerband)
    return df

def fetch_stock_data(stock_symbol):
    df = yf.download(stock_symbol, period='1y', interval='1d')
    if df.empty:
        return None
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_sma(df)
    df = calculate_ema(df)
    df = calculate_bollinger_bands(df)
    df = calculate_adx(df)
    df = calculate_supertrend(df)
    return df

def save_to_excel(dataframes, filename="nse_stock_screener_results.xlsx"):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for stock, df in dataframes.items():
            df.to_excel(writer, sheet_name=stock, index=True)
    print(f"Data saved to {filename}")

nse_stock_list = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]  # Replace with full NSE stock list
stock_data = {}

for stock in nse_stock_list:
    df = fetch_stock_data(stock)
    if df is not None:
        stock_data[stock] = df

save_to_excel(stock_data)
