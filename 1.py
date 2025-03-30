import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import time
import os
import concurrent.futures
import logging
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
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
    try:
        df = pd.read_csv(stock_list_file)
        symbols = [f"{s.strip()}.NS" for s in df['Symbol'].tolist() if s.strip()]
        logger.info(f"[SUCCESS] Fetched {len(symbols)} NSE symbols from '{stock_list_file}'")
        return symbols
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch NSE symbols: {e}")
        return [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "ITC.NS",
            "ICICIBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
            "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "MARUTI.NS", "AXISBANK.NS"
        ]

def fetch_stock_data_parallel(stock_list, period="1y", interval="1d",
                              max_workers=10, batch_size=100, sleep=5, timeout=120):
    all_data = []
    total = len(stock_list)
    success = 0
    failed = 0

    try:
        for i in range(0, total, batch_size):
            batch = stock_list[i:i+batch_size]
            logger.info(f"[INFO] Processing batch {i//batch_size + 1}/{(total+batch_size-1)//batch_size} ({len(batch)} stocks)")
            batch_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_stock, s, period, interval): s for s in batch}
                
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=timeout):
                        stock = futures[future]
                        try:
                            data = future.result()
                            if data:
                                batch_results.append(data)
                                success += 1
                                if success % 10 == 0:
                                    logger.info(f"[INFO] Processed {success}/{total} stocks successfully")
                            else:
                                failed += 1
                        except concurrent.futures.TimeoutError:
                            logger.error(f"[ERROR] Timeout processing {stock}")
                            failed += 1
                        except Exception as e:
                            logger.error(f"[ERROR] Failed to process {stock}: {e}")
                            failed += 1
                except concurrent.futures.TimeoutError:
                    logger.error(f"[ERROR] Batch timeout: {len(futures)-success-failed} stocks skipped")
                    failed += len(futures) - success - failed
            
            all_data.extend(batch_results)
            if i + batch_size < total:
                logger.info(f"[INFO] Pausing for {sleep} seconds to avoid API limits...")
                time.sleep(sleep)
        
        logger.info(f"[INFO] Total: {success} stocks processed, {failed} failed")
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
    except KeyboardInterrupt:
        logger.error("[ERROR] Script interrupted. Exiting...")
        return pd.DataFrame()

def process_stock(stock, period, interval):
    try:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty or len(hist) < 50:
            logger.warning(f"[WARNING] Insufficient data for {stock} ({len(hist)} days)")
            return None
        
        required_cols = ['High', 'Low', 'Close', 'Open', 'Volume']
        if not all(c in hist.columns for c in required_cols):
            logger.warning(f"[WARNING] Missing columns for {stock}: {list(hist.columns)}")
            return None
        
        hist = calculate_indicators(hist, stock)
        
        # Validate all critical columns exist
        critical_cols = [
            'RSI', 'MACD', 'Supertrend', 'Supertrend_Direction',
            '52_Week_High', '52_Week_Low', 'Resistance', 'Support',  # Added Resistance/Support
            'Volume_SMA_20', 'BB_Upper', 'BB_Lower'  # Added BB and Volume columns
        ]
        if any(col not in hist.columns for col in critical_cols):
            logger.warning(f"[WARNING] Missing indicators for {stock}: {critical_cols}")
            return None
        
        # Skip if NaN in critical columns
        if hist[critical_cols].isnull().any().any():
            logger.warning(f"[WARNING] NaN values detected in indicators for {stock}")
            return None
        
        return extract_latest_data(hist, stock)
    except Exception as e:
        logger.error(f"[ERROR] Error processing {stock}: {e}")
        return None

def calculate_indicators(hist, stock_symbol):
    try:
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        macd = ta.macd(hist['Close'], fast=12, slow=26, signal=9)
        hist['MACD'] = macd['MACD_12_26_9']
        hist['MACD_Signal'] = macd['MACDs_12_26_9']
        hist['MACD_Hist'] = macd['MACDh_12_26_9']
        hist['SMA_50'] = ta.sma(hist['Close'], length=50)
        hist['SMA_200'] = ta.sma(hist['Close'], length=200)
        hist['EMA_50'] = ta.ema(hist['Close'], length=50)
        hist['EMA_200'] = ta.ema(hist['Close'], length=200)
        bb = ta.bbands(hist['Close'], length=20, std=2)
        hist['BB_Upper'] = bb['BBU_20_2.0']
        hist['BB_Lower'] = bb['BBL_20_2.0']
        hist['BB_Middle'] = bb['BBM_20_2.0']
        hist['ADX'] = ta.adx(hist['High'], hist['Low'], hist['Close'], length=14)['ADX_14']
        hist['ATR'] = ta.atr(hist['High'], hist['Low'], hist['Close'], length=14).fillna(0)
        
        # Supertrend with version handling
        supertrend = ta.supertrend(hist['High'], hist['Low'], hist['Close'], length=10, multiplier=4)
        if supertrend is not None:
            if 'SUPERT_10_4.0' in supertrend.columns:
                hist['Supertrend'] = supertrend['SUPERT_10_4.0']
                hist['Supertrend_Direction'] = supertrend['SUPERTd_10_4.0']
            elif 'SUPERT_10_4' in supertrend.columns:
                hist['Supertrend'] = supertrend['SUPERT_10_4']
                hist['Supertrend_Direction'] = supertrend['SUPERTd_10_4']
            else:
                hist['Supertrend'] = None
                hist['Supertrend_Direction'] = None
                logger.warning(f"[WARNING] Supertrend columns not found for {stock_symbol}")
        else:
            hist['Supertrend'] = None
            hist['Supertrend_Direction'] = None
            logger.warning(f"[WARNING] Supertrend failed for {stock_symbol}")
        
        # 52-week high/low with fallback
        try:
            hist['52_Week_High'] = hist['Close'].rolling(252, min_periods=1).max()
            hist['52_Week_Low'] = hist['Close'].rolling(252, min_periods=1).min()
        except Exception as e:
            logger.warning(f"[WARNING] 52-week calculation failed for {stock_symbol}: {e}")
            hist['52_Week_High'] = hist['Close']
            hist['52_Week_Low'] = hist['Close']
        
        # Support/Resistance with safety
        try:
            hist['Support'] = hist['Close'].rolling(50, min_periods=1).min()
            hist['Resistance'] = hist['Close'].rolling(50, min_periods=1).max()
        except Exception as e:
            logger.warning(f"[WARNING] Support/Resistance failed for {stock_symbol}: {e}")
            hist['Support'] = hist['Close']
            hist['Resistance'] = hist['Close']
        
        hist['Volume_SMA_20'] = hist['Volume'].rolling(20, min_periods=1).mean()
        hist['Volume_Spike'] = hist['Volume'] > hist['Volume_SMA_20'] * 2
        
        # Gap calculations with NaN protection
        hist['Gap_Up'] = (hist['Open'] > hist['Close'].shift(1)) & ((hist['Open'] - hist['Close'].shift(1)) > 0.5 * hist['ATR'])
        hist['Gap_Down'] = (hist['Open'] < hist['Close'].shift(1)) & ((hist['Close'].shift(1) - hist['Open']) > 0.5 * hist['ATR'])
        
        # Price changes
        hist['1d_Change (%)'] = hist['Close'].pct_change(periods=1, fill_method=None) * 100
        hist['1w_Change (%)'] = hist['Close'].pct_change(periods=5, fill_method=None) * 100
        hist['1m_Change (%)'] = hist['Close'].pct_change(periods=20, fill_method=None) * 100
        
        # Stochastic Oscillator
        stoch = ta.stoch(hist['High'], hist['Low'], hist['Close'])
        hist['Stoch_K'] = stoch['STOCHk_14_3_3']
        hist['Stoch_D'] = stoch['STOCHd_14_3_3']
        
        # Williams %R
        hist['Williams_%R'] = ta.willr(hist['High'], hist['Low'], hist['Close'], length=14)
        
        # Chaikin Money Flow
        hist['CMF'] = ta.cmf(hist['High'], hist['Low'], hist['Close'], hist['Volume'], length=20)
        
        # Forward/backward fill NaN values
        return hist.ffill().bfill()
    except Exception as e:
        logger.error(f"[ERROR] Indicator failure for {stock_symbol}: {e}")
        return hist

def extract_latest_data(hist, stock_symbol):
    latest = hist.iloc[-1]
    data = {
        "Stock": stock_symbol,
        "Date": hist.index[-1].strftime('%Y-%m-%d'),
        "Price (₹)": latest.get('Close', 0),
        "RSI": latest.get('RSI'),
        "MACD": latest.get('MACD'),
        "MACD_Signal": latest.get('MACD_Signal'),
        "MACD_Hist": latest.get('MACD_Hist'),
        "MACD_Trend": "Bullish" if latest.get('MACD', 0) > latest.get('MACD_Signal', 0) else "Bearish",
        "SMA_50": latest.get('SMA_50'),
        "SMA_200": latest.get('SMA_200'),
        "SMA_Trend": "Bullish" if latest.get('SMA_50', 0) > latest.get('SMA_200', 0) else "Bearish",
        "EMA_50": latest.get('EMA_50'),
        "EMA_200": latest.get('EMA_200'),
        "EMA_Trend": "Bullish" if latest.get('EMA_50', 0) > latest.get('EMA_200', 0) else "Bearish",
        "BB_Upper": latest.get('BB_Upper'),
        "BB_Lower": latest.get('BB_Lower'),
        "BB_Middle": latest.get('BB_Middle'),
        "BB_Position": "Upper" if latest.get('Close', 0) >= latest.get('BB_Upper', 0) else 
                       "Lower" if latest.get('Close', 0) <= latest.get('BB_Lower', 0) else "Middle",
        "ADX": latest.get('ADX'),
        "ATR": latest.get('ATR'),
        "Supertrend": latest.get('Supertrend'),
        "Supertrend_Signal": "Buy" if pd.notnull(latest.get('Supertrend_Direction')) and latest.get('Supertrend_Direction') == 1 else "Sell",
        "Support (₹)": latest.get('Support', 0),  # Added default 0
        "Resistance (₹)": latest.get('Resistance', 0),  # Added default 0
        "Volume": latest.get('Volume'),
        "Volume_SMA_20": latest.get('Volume_SMA_20', 0),  # Added default 0
        "Volume_Spike": latest.get('Volume_Spike', False),
        "52_Week_High (₹)": latest.get('52_Week_High', 0),  # Added default 0
        "52_Week_Low (₹)": latest.get('52_Week_Low', 0),  # Added default 0
        "52_Week_High_Breakout": False,
        "52_Week_Low_Breakout": False,
        "Gap_Up": latest.get('Gap_Up', False),
        "Gap_Down": latest.get('Gap_Down', False),
        "1d_Change (%)": latest.get('1d_Change (%)'),
        "1w_Change (%)": latest.get('1w_Change (%)'),
        "1m_Change (%)": latest.get('1m_Change (%)'),
        "Stoch_K": latest.get('Stoch_K'),
        "Stoch_D": latest.get('Stoch_D'),
        "Williams_%R": latest.get('Williams_%R'),
        "CMF": latest.get('CMF')
    }
    
    # 52-week breakout with safety
    if len(hist) > 252 and '52_Week_High' in hist.columns and '52_Week_Low' in hist.columns:
        data["52_Week_High_Breakout"] = latest['Close'] >= hist['52_Week_High'].iloc[-2]
        data["52_Week_Low_Breakout"] = latest['Close'] <= hist['52_Week_Low'].iloc[-2]
    
    return data

# Filters with strict NaN checks
def oversold(df, threshold=30):
    return (df['RSI'] < threshold) & df['RSI'].notnull()

def overbought(df, threshold=70):
    return (df['RSI'] > threshold) & df['RSI'].notnull()

def golden_cross(df):
    return (
        (df['SMA_Trend'] == "Bullish") 
        & (df['1d_Change (%)'] > 0) 
        & df[['SMA_50', 'SMA_200']].notnull().all(axis=1)
    )

def death_cross(df):
    return (
        (df['SMA_Trend'] == "Bearish") 
        & (df['1d_Change (%)'] < 0) 
        & df[['SMA_50', 'SMA_200']].notnull().all(axis=1)
    )

def macd_bullish(df):
    return (
        (df['MACD_Trend'] == "Bullish") 
        & (df['MACD_Hist'] > 0) 
        & df[['MACD', 'MACD_Signal']].notnull().all(axis=1)
    )

def macd_bearish(df):
    return (
        (df['MACD_Trend'] == "Bearish") 
        & (df['MACD_Hist'] < 0) 
        & df[['MACD', 'MACD_Signal']].notnull().all(axis=1)
    )

def supertrend_buy(df):
    return (df['Supertrend_Signal'] == "Buy") & df['Supertrend_Signal'].notnull()

def supertrend_sell(df):
    return (df['Supertrend_Signal'] == "Sell") & df['Supertrend_Signal'].notnull()

def above_resistance(df):
    return (
        (df['Price (₹)'] > df['Resistance (₹)']) 
        & df[['Price (₹)', 'Resistance (₹)']].notnull().all(axis=1)  # Added column existence check
    )

def below_support(df):
    return (
        (df['Price (₹)'] < df['Support (₹)']) 
        & df[['Price (₹)', 'Support (₹)']].notnull().all(axis=1)  # Added column existence check
    )

def volume_spike(df):
    return df['Volume_Spike'] & df[['Volume_Spike', 'Volume_SMA_20']].notnull().all(axis=1)

def bullish_engulfing(df):
    return (
        (df['1d_Change (%)'] > 1.5) 
        & ~df['Gap_Down'] 
        & df['RSI'].notnull() 
        & (df['RSI'] < 60)
    )

def save_professional_excel(df, output_dir, timestamp):
    excel_file = f"{output_dir}/filtered_stocks_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
        currency_fmt = workbook.add_format({'num_format': '₹#,##0.00'})
        percent_fmt = workbook.add_format({'num_format': '0.00%'})
        number_fmt = workbook.add_format({'num_format': '#,##0.00'})
        
        # Summary sheet
        summary = pd.DataFrame({
            "Filter": [f.replace("_", " ").title() for f in filters.keys()],
            "Count": [len(df[func(df)]) for func in filters.values()]
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)
        ws_summary = writer.sheets["Summary"]
        ws_summary.set_column('A:B', 25, header_fmt)
        
        # Add bar chart
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({'categories': '=Summary!$A$2:$A$13', 'values': '=Summary!$B$2:$B$13'})
        chart.set_title({'name': 'Filter Results Overview'})
        ws_summary.insert_chart('D2', chart)
        
        # Filter sheets
        for name, func in filters.items():
            filtered = df[func(df)]
            if not filtered.empty:
                sheet_name = name.replace("_", " ").title()[:31]
                filtered.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                
                # Formatting
                ws.set_column('A:A', 15)
                ws.set_column('B:B', 12, currency_fmt)
                ws.set_column('C:C', 10, percent_fmt)
                ws.set_column('D:F', 15, number_fmt)
                ws.set_column('G:G', 12)
                ws.freeze_panes(1, 0)
                
                # RSI formatting
                if 'RSI' in filtered.columns:
                    rsi_col = filtered.columns.get_loc("RSI") + 1
                    ws.conditional_format(1, rsi_col, len(filtered), rsi_col, {
                        'type': 'cell', 'criteria': '>=', 'value': 70, 'format': workbook.add_format({'bg_color': '#FF9999'})
                    })
                    ws.conditional_format(1, rsi_col, len(filtered), rsi_col, {
                        'type': 'cell', 'criteria': '<=', 'value': 30, 'format': workbook.add_format({'bg_color': '#99FF99'})
                    })
                
                # Price vs SMA 50 chart
                if 'SMA_50' in filtered.columns and len(filtered) > 1:
                    chart = workbook.add_chart({'type': 'line'})
                    chart.add_series({'name': 'Price', 'values': f"='{sheet_name}'!B2:B{len(filtered)+1}"})
                    chart.add_series({'name': 'SMA 50', 'values': f"='{sheet_name}'!H2:H{len(filtered)+1}"})
                    chart.set_title({'name': 'Price vs SMA 50'})
                    ws.insert_chart('K2', chart)

def run_screener(stock_list_file, period="1y",
                max_workers=10, batch_size=100, sleep=5, timeout=120):
    start_time = time.time()
    output_dir = "stock_screener_results"
    os.makedirs(output_dir, exist_ok=True)
    
    stock_list = fetch_all_nse_symbols(stock_list_file)
    
    try:
        df = fetch_stock_data_parallel(
            stock_list,
            period=period,
            max_workers=max_workers,
            batch_size=batch_size,
            sleep=sleep,
            timeout=timeout
        )
    except KeyboardInterrupt:
        logger.error("[ERROR] Script interrupted. Exiting...")
        sys.exit(1)
    
    if df.empty:
        logger.error("[ERROR] No data fetched. Exiting.")
        return
    
    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"{output_dir}/all_stocks_data_{timestamp}.csv", index=False)
    logger.info(f"[INFO] Raw data saved to 'all_stocks_data_{timestamp}.csv'")
    
    # Apply filters
    filters = {
        "oversold": oversold,
        "overbought": overbought,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "macd_bullish": macd_bullish,
        "macd_bearish": macd_bearish,
        "supertrend_buy": supertrend_buy,
        "supertrend_sell": supertrend_sell,
        "above_resistance": above_resistance,
        "below_support": below_support,
        "volume_spike": volume_spike,
        "bullish_engulfing": bullish_engulfing
    }
    
    save_professional_excel(df, output_dir, timestamp)
    logger.info(f"[INFO] Professional Excel saved to 'filtered_stocks_{timestamp}.xlsx'")
    
    # Log execution time
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    logger.info(f"[INFO] Execution time: {int(hours)}h {int(mins)}m {secs:.2f}s")

if __name__ == "__main__":
    try:
        run_screener(
            stock_list_file="nse_stock_list.csv",
            period="1y",
            max_workers=10,
            batch_size=100,
            sleep=5,
            timeout=120
        )
    except KeyboardInterrupt:
        logger.error("[ERROR] User terminated the script.")
        sys.exit(1)