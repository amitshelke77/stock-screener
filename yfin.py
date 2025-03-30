import yfinance as yf
import pandas as pd
import pandas_ta as ta

# List of NSE stock symbols (Yahoo Finance format: 'RELIANCE.NS' for NSE stocks)
stock_list = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "ITC.NS", "ICICIBANK.NS", "HINDUNILVR.NS"]
data = []

# Fetch data for each stock
for stock in stock_list:
    try:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period="1y")  # Get 1 year of historical data

        if hist.empty or len(hist) < 50:  # Ensure enough data
            print(f"❌ Not enough data for {stock} (only {len(hist)} days)")
            continue

        latest_price = hist["Close"].iloc[-1]  # Latest closing price

        # Calculate Indicators
        hist['RSI'] = ta.rsi(hist["Close"], length=14)
        macd = ta.macd(hist["Close"])
        hist['MACD'] = macd['MACD_12_26_9']
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
        hist['Supertrend'] = ta.supertrend(hist["High"], hist["Low"], hist["Close"], length=10, multiplier=4)['SUPERT_10_4.0']

        # Support & Resistance Levels
        hist['Support'] = hist['Close'].rolling(50).min()
        hist['Resistance'] = hist['Close'].rolling(50).max()

        # Volume Analysis
        hist['Volume_Spike'] = hist['Volume'] > hist['Volume'].rolling(10).mean() * 2

        # 52-Week High & Low Breakouts
        hist['52_Week_High'] = hist['Close'].rolling(252).max()
        hist['52_Week_Low'] = hist['Close'].rolling(252).min()
        is_breaking_high = latest_price >= hist['52_Week_High'].iloc[-1]
        is_breaking_low = latest_price <= hist['52_Week_Low'].iloc[-1]

        # Gap Up / Gap Down Stocks
        hist['Gap_Up'] = (hist['Open'] > hist['Close'].shift(1)) & (hist['Open'] - hist['Close'].shift(1) > 0.5 * hist['ATR'])
        hist['Gap_Down'] = (hist['Open'] < hist['Close'].shift(1)) & (hist['Close'].shift(1) - hist['Open'] > 0.5 * hist['ATR'])

        latest_data = {
            "Stock": stock, "Price": latest_price, "RSI": hist["RSI"].iloc[-1], "MACD": hist["MACD"].iloc[-1],
            "SMA_50": hist['SMA_50'].iloc[-1], "SMA_200": hist['SMA_200'].iloc[-1], "EMA_50": hist['EMA_50'].iloc[-1], "EMA_200": hist['EMA_200'].iloc[-1],
            "BB_upper": hist['BB_upper'].iloc[-1], "BB_lower": hist['BB_lower'].iloc[-1], "BB_middle": hist['BB_middle'].iloc[-1],
            "ADX": hist['ADX'].iloc[-1], "ATR": hist['ATR'].iloc[-1], "Supertrend": hist['Supertrend'].iloc[-1],
            "Support": hist['Support'].iloc[-1], "Resistance": hist['Resistance'].iloc[-1],
            "Volume_Spike": hist['Volume_Spike'].iloc[-1], "52_Week_High_Breakout": is_breaking_high, "52_Week_Low_Breakout": is_breaking_low,
            "Gap_Up": hist['Gap_Up'].iloc[-1], "Gap_Down": hist['Gap_Down'].iloc[-1]
        }

        data.append(latest_data)
        print(f"✅ {stock}: ₹{latest_price:.2f} | RSI: {latest_data['RSI']:.2f} | MACD: {latest_data['MACD']:.2f} | Supertrend: {latest_data['Supertrend']:.2f}")

    except Exception as e:
        print(f"❌ Error fetching data for {stock}: {e}")

# Convert to DataFrame
df = pd.DataFrame(data)

# Identify Oversold & Overbought Stocks
oversold_stocks = df[df['RSI'] < 30]
overbought_stocks = df[df['RSI'] > 70]

# Display Results
print("\n🔴 Oversold Stocks (RSI < 30):")
print(oversold_stocks[['Stock', 'Price', 'RSI']])
print("\n🟢 Overbought Stocks (RSI > 70):")
print(overbought_stocks[['Stock', 'Price', 'RSI']])

# Save to CSV
df.to_csv("stock_screener_results.csv", index=False)
print("\n📁 Data saved to 'stock_screener_results.csv'")
