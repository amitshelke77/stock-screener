import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
import time
import os
from nsetools import Nse

def fetch_all_nse_symbols():
    """Fetch all NSE stock symbols using nsetools"""
    try:
        nse = Nse()
        all_stock_codes = nse.get_stock_codes()
        # Convert to Yahoo Finance format (add .NS suffix)
        symbols = [f"{code}.NS" for code in all_stock_codes.keys() if code != 'SYMBOL']
        print(f"✅ Fetched {len(symbols)} NSE stock symbols")
        return symbols
    except Exception as e:
        print(f"❌ Error fetching NSE symbols: {e}")
        # Fallback to predefined list if nsetools fails
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "ITC.NS", 
                "ICICIBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
                "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "MARUTI.NS", "AXISBANK.NS"]

def fetch_stock_data(stock_list, period="1y", interval="1d", batch_size=50, 
                     sleep_between_batches=2):
    """
    Fetch data for a list of stocks with error handling and batching
    
    Parameters:
    - stock_list: List of stock symbols
    - period: Time period for historical data
    - interval: Data interval (1d, 1h, etc.)
    - batch_size: Number of stocks to process in each batch
    - sleep_between_batches: Seconds to pause between batches to avoid rate limits
    
    Returns:
    - DataFrame with stock data and calculated indicators
    """
    all_data = []
    total_stocks = len(stock_list)
    
    # Process stocks in batches
    for i in range(0, total_stocks, batch_size):
        batch = stock_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_stocks+batch_size-1)//batch_size} ({len(batch)} stocks)")
        
        for stock in batch:
            try:
                ticker = yf.Ticker(stock)
                hist = ticker.history(period=period, interval=interval)
                
                if hist.empty or len(hist) < 50:
                    print(f"❌ Not enough data for {stock} (only {len(hist)} days)")
                    continue
                
                latest_price = hist["Close"].iloc[-1]
                
                # Calculate all indicators
                calculate_indicators(hist)
                
                # Prepare stock data
                latest_data = extract_latest_data(hist, stock)
                all_data.append(latest_data)
                
                print(f"✅ {stock}: ₹{latest_price:.2f} | RSI: {latest_data['RSI']:.2f} | MACD: {latest_data['MACD']:.2f}")
                
            except Exception as e:
                print(f"❌ Error fetching data for {stock}: {e}")
        
        # Sleep between batches to avoid rate limiting
        if i + batch_size < total_stocks:
            print(f"Pausing for {sleep_between_batches} seconds to avoid rate limiting...")
            time.sleep(sleep_between_batches)
    
    # Convert to DataFrame
    if all_data:
        return pd.DataFrame(all_data)
    else:
        print("No data could be fetched for any stocks.")
        return pd.DataFrame()

def calculate_indicators(hist):
    """Calculate technical indicators for the given historical data"""
    # RSI
    hist['RSI'] = ta.rsi(hist["Close"], length=14)
    
    # MACD
    macd = ta.macd(hist["Close"])
    hist['MACD'] = macd['MACD_12_26_9']
    hist['MACD_Signal'] = macd['MACDs_12_26_9']
    hist['MACD_Hist'] = macd['MACDh_12_26_9']
    
    # Moving Averages
    hist['SMA_50'] = ta.sma(hist["Close"], length=50)
    hist['SMA_200'] = ta.sma(hist["Close"], length=200)
    hist['EMA_50'] = ta.ema(hist["Close"], length=50)
    hist['EMA_200'] = ta.ema(hist["Close"], length=200)
    
    # Bollinger Bands
    bb = ta.bbands(hist["Close"], length=20)
    hist['BB_upper'] = bb['BBU_20_2.0']
    hist['BB_lower'] = bb['BBL_20_2.0']
    hist['BB_middle'] = bb['BBM_20_2.0']
    
    # Other Indicators
    hist['ADX'] = ta.adx(hist["High"], hist["Low"], hist["Close"], length=14)['ADX_14']
    hist['ATR'] = ta.atr(hist["High"], hist["Low"], hist["Close"], length=14)
    
    # Supertrend
    supertrend = ta.supertrend(hist["High"], hist["Low"], hist["Close"], length=10, multiplier=4)
    hist['Supertrend'] = supertrend['SUPERT_10_4.0']
    hist['Supertrend_Direction'] = supertrend['SUPERTd_10_4.0']
    
    # Support & Resistance Levels
    hist['Support'] = hist['Close'].rolling(50).min()
    hist['Resistance'] = hist['Close'].rolling(50).max()
    
    # Volume Analysis
    hist['Volume_SMA_20'] = hist['Volume'].rolling(20).mean()
    hist['Volume_Spike'] = hist['Volume'] > hist['Volume_SMA_20'] * 2
    
    # 52-Week High & Low
    hist['52_Week_High'] = hist['Close'].rolling(252).max()
    hist['52_Week_Low'] = hist['Close'].rolling(252).min()
    
    # Gap Analysis
    hist['Gap_Up'] = (hist['Open'] > hist['Close'].shift(1)) & (hist['Open'] - hist['Close'].shift(1) > 0.5 * hist['ATR'])
    hist['Gap_Down'] = (hist['Open'] < hist['Close'].shift(1)) & (hist['Close'].shift(1) - hist['Open'] > 0.5 * hist['ATR'])
    
    # Price change percentages
    hist['1d_Change_Pct'] = hist['Close'].pct_change(1) * 100
    hist['1w_Change_Pct'] = hist['Close'].pct_change(5) * 100
    hist['1m_Change_Pct'] = hist['Close'].pct_change(20) * 100
    
    return hist

def extract_latest_data(hist, stock_symbol):
    """Extract the latest data for a stock"""
    latest_price = hist["Close"].iloc[-1]
    latest_date = hist.index[-1]
    
    # Check for breakouts
    is_breaking_high = latest_price >= hist['52_Week_High'].iloc[-2] if len(hist) > 252 else False
    is_breaking_low = latest_price <= hist['52_Week_Low'].iloc[-2] if len(hist) > 252 else False
    
    # Trend determination
    sma_trend = "Bullish" if hist['SMA_50'].iloc[-1] > hist['SMA_200'].iloc[-1] else "Bearish"
    ema_trend = "Bullish" if hist['EMA_50'].iloc[-1] > hist['EMA_200'].iloc[-1] else "Bearish"
    
    # MACD signal
    macd_signal = "Bullish" if hist['MACD'].iloc[-1] > hist['MACD_Signal'].iloc[-1] else "Bearish"
    
    # Supertrend signal
    supertrend_signal = "Buy" if hist['Supertrend_Direction'].iloc[-1] == 1 else "Sell"
    
    # Bollinger Band position
    bb_position = "Upper" if latest_price >= hist['BB_upper'].iloc[-1] else \
                 "Lower" if latest_price <= hist['BB_lower'].iloc[-1] else "Middle"
    
    return {
        "Stock": stock_symbol,
        "Date": latest_date.strftime('%Y-%m-%d') if isinstance(latest_date, datetime) else str(latest_date),
        "Price": latest_price,
        
        # Technical indicators
        "RSI": hist["RSI"].iloc[-1],
        "MACD": hist["MACD"].iloc[-1],
        "MACD_Signal": hist["MACD_Signal"].iloc[-1],
        "MACD_Hist": hist["MACD_Hist"].iloc[-1],
        "MACD_Trend": macd_signal,
        
        "SMA_50": hist['SMA_50'].iloc[-1],
        "SMA_200": hist['SMA_200'].iloc[-1],
        "SMA_Trend": sma_trend,
        
        "EMA_50": hist['EMA_50'].iloc[-1],
        "EMA_200": hist['EMA_200'].iloc[-1],
        "EMA_Trend": ema_trend,
        
        "BB_upper": hist['BB_upper'].iloc[-1],
        "BB_lower": hist['BB_lower'].iloc[-1],
        "BB_middle": hist['BB_middle'].iloc[-1],
        "BB_Position": bb_position,
        
        "ADX": hist['ADX'].iloc[-1],
        "ATR": hist['ATR'].iloc[-1],
        
        "Supertrend": hist['Supertrend'].iloc[-1],
        "Supertrend_Signal": supertrend_signal,
        
        "Support": hist['Support'].iloc[-1],
        "Resistance": hist['Resistance'].iloc[-1],
        
        "Volume": hist['Volume'].iloc[-1],
        "Volume_SMA_20": hist['Volume_SMA_20'].iloc[-1],
        "Volume_Spike": bool(hist['Volume_Spike'].iloc[-1]),
        
        "52_Week_High": hist['52_Week_High'].iloc[-1],
        "52_Week_Low": hist['52_Week_Low'].iloc[-1],
        "52_Week_High_Breakout": is_breaking_high,
        "52_Week_Low_Breakout": is_breaking_low,
        
        "Gap_Up": bool(hist['Gap_Up'].iloc[-1]),
        "Gap_Down": bool(hist['Gap_Down'].iloc[-1]),
        
        "1d_Change_Pct": hist['1d_Change_Pct'].iloc[-1],
        "1w_Change_Pct": hist['1w_Change_Pct'].iloc[-1],
        "1m_Change_Pct": hist['1m_Change_Pct'].iloc[-1],
    }

# Filter functions
def filter_stocks(df, filter_func):
    """Filter stocks based on a filter function"""
    return df[filter_func(df)]

# Predefined filter functions
def oversold_stocks(df, rsi_threshold=30):
    """Filter for oversold stocks based on RSI"""
    return df['RSI'] < rsi_threshold

def overbought_stocks(df, rsi_threshold=70):
    """Filter for overbought stocks based on RSI"""
    return df['RSI'] > rsi_threshold

def golden_cross(df):
    """Filter for stocks with recent golden cross (SMA50 crosses above SMA200)"""
    return (df['SMA_Trend'] == "Bullish") & (df['1d_Change_Pct'] > 0)

def death_cross(df):
    """Filter for stocks with recent death cross (SMA50 crosses below SMA200)"""
    return (df['SMA_Trend'] == "Bearish") & (df['1d_Change_Pct'] < 0)

def macd_bullish_crossover(df):
    """Filter for stocks with MACD crossing above signal line"""
    return (df['MACD_Trend'] == "Bullish") & (df['MACD_Hist'] > 0)

def macd_bearish_crossover(df):
    """Filter for stocks with MACD crossing below signal line"""
    return (df['MACD_Trend'] == "Bearish") & (df['MACD_Hist'] < 0)

def supertrend_buy_signal(df):
    """Filter for stocks with Supertrend buy signal"""
    return df['Supertrend_Signal'] == "Buy"

def supertrend_sell_signal(df):
    """Filter for stocks with Supertrend sell signal"""
    return df['Supertrend_Signal'] == "Sell"

def above_resistance(df):
    """Filter for stocks breaking above resistance level"""
    return df['Price'] > df['Resistance']

def below_support(df):
    """Filter for stocks breaking below support level"""
    return df['Price'] < df['Support']

def volume_breakout(df):
    """Filter for stocks with volume spike"""
    return df['Volume_Spike'] == True

def bullish_engulfing(df):
    """Custom filter function for bullish engulfing pattern"""
    return (df['1d_Change_Pct'] > 1.5) & (df['Gap_Down'] == False) & (df['RSI'] < 60)

def run_screener(use_all_nse=False, custom_stocks=None, period="1y", 
                 save_all_data=True, apply_filters=True):
    """
    Run the stock screener with various options
    
    Parameters:
    - use_all_nse: Whether to use all NSE stocks
    - custom_stocks: Custom list of stock symbols
    - period: Time period for historical data
    - save_all_data: Whether to save all stock data to CSV
    - apply_filters: Whether to apply predefined filters
    
    Returns:
    - DataFrame with all stock data
    """
    start_time = time.time()
    
    # Determine which stocks to use
    if custom_stocks:
        stock_list = custom_stocks
        print(f"Using custom list of {len(stock_list)} stocks")
    elif use_all_nse:
        stock_list = fetch_all_nse_symbols()
    else:
        stock_list = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "ITC.NS", 
                      "ICICIBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
                      "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "MARUTI.NS", "AXISBANK.NS"]
        print(f"Using default list of {len(stock_list)} stocks")
    
    # Create output directory
    output_dir = "stock_screener_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Fetch stock data
    print(f"\n📊 Fetching data for {len(stock_list)} stocks...")
    df = fetch_stock_data(stock_list, period=period)
    
    if df.empty:
        print("No data was fetched. Exiting.")
        return None
    
    # Save all data
    if save_all_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_data_file = f"{output_dir}/all_stocks_data_{timestamp}.csv"
        df.to_csv(all_data_file, index=False)
        print(f"\n📁 All stock data saved to '{all_data_file}'")
    
    # Apply filters if requested
    if apply_filters:
        # Create dictionary of filter functions and their names
        filters = {
            "oversold": oversold_stocks,
            "overbought": overbought_stocks,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "macd_bullish": macd_bullish_crossover,
            "macd_bearish": macd_bearish_crossover,
            "supertrend_buy": supertrend_buy_signal,
            "supertrend_sell": supertrend_sell_signal,
            "above_resistance": above_resistance,
            "below_support": below_support,
            "volume_breakout": volume_breakout,
            "bullish_engulfing": bullish_engulfing
        }
        
        # Apply each filter and save results
        print("\n🔍 Applying filters...")
        for filter_name, filter_func in filters.items():
            filtered_df = filter_stocks(df, filter_func)
            
            if not filtered_df.empty:
                print(f"\n📋 {filter_name.replace('_', ' ').title()} Stocks: {len(filtered_df)} found")
                print(filtered_df[['Stock', 'Price', 'RSI', 'MACD_Trend', 'Supertrend_Signal']].head())
                
                # Save filtered results
                filter_file = f"{output_dir}/{filter_name}_stocks_{timestamp}.csv"
                filtered_df.to_csv(filter_file, index=False)
                print(f"📁 Saved to '{filter_file}'")
    
    # Print execution time
    end_time = time.time()
    print(f"\n⏱️ Execution time: {end_time - start_time:.2f} seconds")
    
    return df

if __name__ == "__main__":
    # Run the screener with all NSE stocks
    # Uncomment the option you want to use
    
    # Option 1: Run with default stock list
    run_screener(use_all_nse=False, period="1y")
    
    # Option 2: Run with all NSE stocks
    # run_screener(use_all_nse=True, period="1y")
    
    # Option 3: Run with custom stock list
    # custom_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS"]
    # run_screener(custom_stocks=custom_stocks, period="6mo")