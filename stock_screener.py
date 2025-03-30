import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import time
import os
import concurrent.futures
import logging
import sys

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_screener.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def fetch_all_nse_symbols(stock_list_file):
    """Fetch all NSE stock symbols from the provided CSV file"""
    try:
        nse_tickers = pd.read_csv(stock_list_file)
        symbols = [f"{symbol}.NS" for symbol in nse_tickers['Symbol'].tolist()]
        logger.info(f"[SUCCESS] Fetched {len(symbols)} NSE stock symbols from '{stock_list_file}'")
        return symbols
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch NSE symbols: {e}")
        # Fallback to predefined list if fetching fails
        fallback_list = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "ITC.NS",
            "ICICIBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
            "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "MARUTI.NS", "AXISBANK.NS"
        ]
        logger.info(f"[INFO] Using fallback list of {len(fallback_list)} stocks")
        return fallback_list

def fetch_stock_data_parallel(stock_list, period="1y", interval="1d", 
                              max_workers=10, batch_size=100, sleep_between_batches=5):
    """
    Fetch data for a list of stocks with parallel processing for improved performance
    """
    all_data = []
    total_stocks = len(stock_list)
    successful_fetches = 0
    failed_fetches = 0

    for i in range(0, total_stocks, batch_size):
        batch = stock_list[i:i+batch_size]
        logger.info(f"[INFO] Processing batch {i//batch_size + 1}/{(total_stocks+batch_size-1)//batch_size} ({len(batch)} stocks)")
        batch_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(process_single_stock, stock, period, interval): stock 
                for stock in batch
            }
            
            for future in concurrent.futures.as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        successful_fetches += 1
                        if successful_fetches % 10 == 0:
                            logger.info(f"[INFO] Successfully processed {successful_fetches} stocks so far")
                    else:
                        failed_fetches += 1
                except Exception as e:
                    logger.error(f"[ERROR] Exception while processing {stock}: {str(e)}")
                    failed_fetches += 1
        
        all_data.extend(batch_results)
        
        if i + batch_size < total_stocks:
            logger.info(f"[INFO] Completed batch with {len(batch_results)} successful fetches. Pausing for {sleep_between_batches} seconds...")
            time.sleep(sleep_between_batches)
    
    if all_data:
        logger.info(f"[INFO] Total: {successful_fetches} stocks processed successfully, {failed_fetches} failed")
        return pd.DataFrame(all_data)
    else:
        logger.error("[ERROR] No data could be fetched for any stocks.")
        return pd.DataFrame()

def process_single_stock(stock, period, interval):
    """Process a single stock and return the extracted data"""
    try:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty or len(hist) < 50:
            logger.warning(f"[WARNING] Not enough data for {stock} (only {len(hist)} days)")
            return None
        
        required_columns = ['High', 'Low', 'Close', 'Open', 'Volume']
        if not all(col in hist.columns for col in required_columns):
            logger.warning(f"[WARNING] Missing required columns for {stock}. Skipping...")
            return None
        
        latest_price = hist["Close"].iloc[-1]
        hist = calculate_indicators(hist)
        latest_data = extract_latest_data(hist, stock)
        logger.info(f"[SUCCESS] Processed {stock}: ₹{latest_price:.2f}")
        return latest_data
    except Exception as e:
        logger.error(f"[ERROR] Error fetching data for {stock}: {e}")
        return None

def calculate_indicators(hist):
    """Calculate technical indicators for the given historical data"""
    hist['RSI'] = ta.rsi(hist["Close"], length=14)
    macd = ta.macd(hist["Close"])
    hist['MACD'] = macd['MACD_12_26_9']
    hist['MACD_Signal'] = macd['MACDs_12_26_9']
    hist['MACD_Hist'] = macd['MACDh_12_26_9']
    hist['SMA_50'] = ta.sma(hist["Close"], length=50)
    hist['SMA_200'] = ta.sma(hist["Close"], length=200)
    hist['EMA_50'] = ta.ema(hist["Close"], length=50)
    hist['EMA_200'] = ta.ema(hist["Close"], length=200)
    bb = ta.bbands(hist["Close"], length=20)
    hist['BB_upper'] = bb['BBU_20_2.0']
    hist['BB_lower'] = bb['BBL_20_2.0']
    hist['BB_middle'] = bb['BBM_20_2.0']
    hist['ADX'] = ta.adx(hist["High"], hist["Low"], hist["Close"], length=14)['ADX_14']
    hist['ATR'] = ta.atr(hist["High"], hist["Low"], hist["Close"], length=14)
    supertrend = ta.supertrend(hist["High"], hist["Low"], hist["Close"], length=10, multiplier=4)
    hist['Supertrend'] = supertrend['SUPERT_10_4.0']
    hist['Supertrend_Direction'] = supertrend['SUPERTd_10_4.0']
    hist['Support'] = hist['Close'].rolling(50).min()
    hist['Resistance'] = hist['Close'].rolling(50).max()
    hist['Volume_SMA_20'] = hist['Volume'].rolling(20).mean()
    hist['Volume_Spike'] = hist['Volume'] > hist['Volume_SMA_20'] * 2
    hist['52_Week_High'] = hist['Close'].rolling(252).max()
    hist['52_Week_Low'] = hist['Close'].rolling(252).min()
    hist['Gap_Up'] = (hist['Open'] > hist['Close'].shift(1)) & (hist['Open'] - hist['Close'].shift(1) > 0.5 * hist['ATR'])
    hist['Gap_Down'] = (hist['Open'] < hist['Close'].shift(1)) & (hist['Close'].shift(1) - hist['Open'] > 0.5 * hist['ATR'])
    hist['1d_Change_Pct'] = hist['Close'].pct_change(1) * 100
    hist['1w_Change_Pct'] = hist['Close'].pct_change(5) * 100
    hist['1m_Change_Pct'] = hist['Close'].pct_change(20) * 100
    
    # Additional Indicators
    stoch = ta.stoch(hist["High"], hist["Low"], hist["Close"])
    hist['Stoch_K'] = stoch['STOCHk_14_3_3']
    hist['Stoch_D'] = stoch['STOCHd_14_3_3']
    hist['Williams_%R'] = ta.willr(hist["High"], hist["Low"], hist["Close"], length=14)
    hist['CMF'] = ta.cmf(hist["High"], hist["Low"], hist["Close"], hist["Volume"], length=20)
    return hist

def extract_latest_data(hist, stock_symbol):
    """Extract the latest data for a stock"""
    latest_price = hist["Close"].iloc[-1]
    latest_date = hist.index[-1]
    is_breaking_high = latest_price >= hist['52_Week_High'].iloc[-2] if len(hist) > 252 else False
    is_breaking_low = latest_price <= hist['52_Week_Low'].iloc[-2] if len(hist) > 252 else False
    sma_trend = "Bullish" if hist['SMA_50'].iloc[-1] > hist['SMA_200'].iloc[-1] else "Bearish"
    ema_trend = "Bullish" if hist['EMA_50'].iloc[-1] > hist['EMA_200'].iloc[-1] else "Bearish"
    macd_signal = "Bullish" if hist['MACD'].iloc[-1] > hist['MACD_Signal'].iloc[-1] else "Bearish"
    supertrend_signal = "Buy" if hist['Supertrend_Direction'].iloc[-1] == 1 else "Sell"
    bb_position = "Upper" if latest_price >= hist['BB_upper'].iloc[-1] else \
                 "Lower" if latest_price <= hist['BB_lower'].iloc[-1] else "Middle"
    
    return {
        "Stock": stock_symbol,
        "Date": latest_date.strftime('%Y-%m-%d') if isinstance(latest_date, datetime) else str(latest_date),
        "Price": latest_price,
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
        "Stoch_K": hist['Stoch_K'].iloc[-1],
        "Stoch_D": hist['Stoch_D'].iloc[-1],
        "Williams_%R": hist['Williams_%R'].iloc[-1],
        "CMF": hist['CMF'].iloc[-1]
    }

# Filter functions
def filter_stocks(df, filter_func):
    """Filter stocks based on a filter function"""
    return df[filter_func(df)]

# Predefined filter functions
def oversold_stocks(df, rsi_threshold=30):
    return df['RSI'] < rsi_threshold

def overbought_stocks(df, rsi_threshold=70):
    return df['RSI'] > rsi_threshold

def golden_cross(df):
    return (df['SMA_Trend'] == "Bullish") & (df['1d_Change_Pct'] > 0)

def death_cross(df):
    return (df['SMA_Trend'] == "Bearish") & (df['1d_Change_Pct'] < 0)

def macd_bullish_crossover(df):
    return (df['MACD_Trend'] == "Bullish") & (df['MACD_Hist'] > 0)

def macd_bearish_crossover(df):
    return (df['MACD_Trend'] == "Bearish") & (df['MACD_Hist'] < 0)

def supertrend_buy_signal(df):
    return df['Supertrend_Signal'] == "Buy"

def supertrend_sell_signal(df):
    return df['Supertrend_Signal'] == "Sell"

def above_resistance(df):
    return df['Price'] > df['Resistance']

def below_support(df):
    return df['Price'] < df['Support']

def volume_breakout(df):
    return df['Volume_Spike'] == True

def bullish_engulfing(df):
    return (df['1d_Change_Pct'] > 1.5) & (df['Gap_Down'] == False) & (df['RSI'] < 60)

def save_professional_excel(df, filters, output_dir, timestamp):
    """Save filtered data to a professionally styled Excel file"""
    try:
        # Create an Excel writer object
        excel_file = f"{output_dir}/filtered_stocks_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Add a summary sheet
            summary_df = pd.DataFrame({
                "Filter Name": list(filters.keys()),
                "Number of Stocks": [len(filter_stocks(df, func)) for func in filters.values()]
            })
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            summary_sheet = writer.sheets["Summary"]
            
            # Apply formatting to the summary sheet
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            number_format = workbook.add_format({'num_format': '#,##0'})
            for col_num, value in enumerate(summary_df.columns.values):
                summary_sheet.write(0, col_num, value, header_format)
            summary_sheet.set_column('A:B', 25)
            
            # Add sheets for each filter
            for filter_name, filter_func in filters.items():
                filtered_df = filter_stocks(df, filter_func)
                if not filtered_df.empty:
                    sheet_name = filter_name.replace("_", " ").title()
                    filtered_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    worksheet = writer.sheets[sheet_name[:31]]
                    
                    # Apply formatting to each sheet
                    for col_num, value in enumerate(filtered_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                    worksheet.set_column('A:Z', 15)
                    worksheet.freeze_panes(1, 0)  # Freeze the first row
                    
                    # Conditional formatting for key columns
                    price_col = filtered_df.columns.get_loc("Price") + 1
                    rsi_col = filtered_df.columns.get_loc("RSI") + 1
                    worksheet.conditional_format(1, price_col, len(filtered_df), price_col, 
                                                 {'type': '3_color_scale', 'min_color': '#FF9999', 
                                                  'mid_color': '#FFFF99', 'max_color': '#99FF99'})
                    worksheet.conditional_format(1, rsi_col, len(filtered_df), rsi_col, 
                                                 {'type': 'cell', 'criteria': '>=', 'value': 70, 
                                                  'format': workbook.add_format({'bg_color': '#FF9999'})})
                    worksheet.conditional_format(1, rsi_col, len(filtered_df), rsi_col, 
                                                 {'type': 'cell', 'criteria': '<=', 'value': 30, 
                                                  'format': workbook.add_format({'bg_color': '#99FF99'})})
        
        logger.info(f"[INFO] Professionally styled Excel file saved to '{excel_file}'")
    except Exception as e:
        logger.error(f"[ERROR] Failed to save Excel file: {e}")

def run_screener(stock_list_file, period="1y", 
                 save_all_data=True, apply_filters=True,
                 max_workers=10, batch_size=100, sleep_between_batches=5):
    start_time = time.time()
    
    stock_list = fetch_all_nse_symbols(stock_list_file)
    
    output_dir = "stock_screener_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.info(f"[INFO] Fetching data for {len(stock_list)} stocks...")
    df = fetch_stock_data_parallel(stock_list, period=period, 
                                  max_workers=max_workers, 
                                  batch_size=batch_size,
                                  sleep_between_batches=sleep_between_batches)
    
    if df.empty:
        logger.error("[ERROR] No data was fetched. Exiting.")
        return None
    
    if save_all_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_data_file = f"{output_dir}/all_stocks_data_{timestamp}.csv"
        df.to_csv(all_data_file, index=False)
        logger.info(f"[INFO] All stock data saved to '{all_data_file}'")
    
    if apply_filters:
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
        
        logger.info("[INFO] Applying filters...")
        for filter_name, filter_func in filters.items():
            filtered_df = filter_stocks(df, filter_func)
            if not filtered_df.empty:
                logger.info(f"[INFO] {filter_name.replace('_', ' ').title()} Stocks: {len(filtered_df)} found")
                logger.info(filtered_df[['Stock', 'Price', 'RSI', 'MACD_Trend', 'Supertrend_Signal']].head().to_string())
                
                filter_file = f"{output_dir}/{filter_name}_stocks_{timestamp}.csv"
                filtered_df.to_csv(filter_file, index=False)
                logger.info(f"[INFO] Saved to '{filter_file}'")
        
        # Save professionally styled Excel file
        save_professional_excel(df, filters, output_dir, timestamp)
    
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"[INFO] Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    return df

if __name__ == "__main__":
    run_screener(
        stock_list_file="nse_stock_list.csv",  # Path to the uploaded CSV file
        period="1y",
        max_workers=10,
        batch_size=100,
        sleep_between_batches=5
    )