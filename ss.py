import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

class AIStockScreener:
    def __init__(self, nifty_stocks_path=None, lookback_period=365, prediction_window=5):
        """
        Initialize the AI Stock Screener
        
        Parameters:
        - nifty_stocks_path: Path to CSV containing Nifty stocks symbols
        - lookback_period: Days of historical data to use
        - prediction_window: Days ahead to predict (e.g., 5 days for a week)
        """
        self.lookback_period = lookback_period
        self.prediction_window = prediction_window
        self.model = None
        
        # Load Nifty stocks
        if nifty_stocks_path:
            self.stocks = pd.read_csv(nifty_stocks_path)['Symbol'].tolist()
        else:
            # Default to major Nifty50 stocks
            self.stocks = [
                'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
                'ICICIBANK.NS', 'KOTAKBANK.NS', 'ITC.NS', 'AXISBANK.NS', 'LT.NS',
                'SBIN.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS'
            ]
        
        self.data = {}
    
    def fetch_data(self, update=False):
        """Fetch historical data for all stocks"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period)
        
        if not self.data or update:
            print(f"Fetching data for {len(self.stocks)} stocks...")
            for symbol in self.stocks:
                try:
                    # Add .NS suffix if not present for NSE stocks
                    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                        fetch_symbol = f"{symbol}.NS"
                    else:
                        fetch_symbol = symbol
                    
                    self.data[symbol] = yf.download(
                        fetch_symbol, 
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d')
                    )
                    print(f"Downloaded {symbol} - {len(self.data[symbol])} days")
                except Exception as e:
                    print(f"Error downloading {symbol}: {e}")
            
            # Remove stocks with insufficient data
            self._clean_data()
    
    def _clean_data(self):
        """Remove stocks with insufficient data"""
        to_remove = []
        for symbol, df in self.data.items():
            if len(df) < self.lookback_period / 2:  # Require at least half of the lookback period
                to_remove.append(symbol)
        
        for symbol in to_remove:
            del self.data[symbol]
            print(f"Removed {symbol} due to insufficient data")
    
    def generate_features(self):
        """Generate technical indicators as features for all stocks"""
        for symbol, df in self.data.items():
            # Basic price and volume features
            df['Returns'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Moving averages
            df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            
            # Exponential moving averages
            df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
            df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
            
            # Relative strength index
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_High'] = bollinger.bollinger_hband()
            df['BB_Low'] = bollinger.bollinger_lband()
            df['BB_Mid'] = bollinger.bollinger_mavg()
            df['BB_%B'] = bollinger.bollinger_pband()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Average True Range (volatility)
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            # On-Balance Volume
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            
            # Additional features
            df['Close_to_SMA_200_Ratio'] = df['Close'] / df['SMA_200']
            df['Close_to_SMA_50_Ratio'] = df['Close'] / df['SMA_50']
            df['SMA_20_to_SMA_50_Ratio'] = df['SMA_20'] / df['SMA_50']
            
            # Generate the target variable (future returns)
            df['Future_Return'] = df['Close'].pct_change(self.prediction_window).shift(-self.prediction_window)
            df['Target'] = np.where(df['Future_Return'] > 0.03, 1, 0)  # 3% return threshold for buy signal
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            self.data[symbol] = df
    
    def prepare_training_data(self):
        """Prepare combined dataset for training the model"""
        all_data = []
        
        feature_columns = [
            'Returns', 'Volume_Change',
            'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 
            'EMA_10', 'EMA_20', 'EMA_50',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_High', 'BB_Low', 'BB_Mid', 'BB_%B',
            'Stoch_K', 'Stoch_D', 'ATR', 'OBV',
            'Close_to_SMA_200_Ratio', 'Close_to_SMA_50_Ratio', 'SMA_20_to_SMA_50_Ratio'
        ]
        
        for symbol, df in self.data.items():
            if len(df) > 0:
                # Extract features and target
                X = df[feature_columns].copy()
                y = df['Target'].copy()
                
                # Add the symbol as a feature (one-hot encoding could be better)
                X['Symbol'] = symbol
                
                # Add to combined dataset
                stock_data = pd.concat([X, y], axis=1)
                all_data.append(stock_data)
        
        # Combine all stock data
        combined_data = pd.concat(all_data, axis=0)
        
        # Convert symbol to numerical
        combined_data['Symbol'] = pd.factorize(combined_data['Symbol'])[0]
        
        # Split features and target
        X = combined_data.drop('Target', axis=1)
        y = combined_data['Target']
        
        return X, y
    
    def train_model(self):
        """Train the predictive model"""
        X, y = self.prepare_training_data()
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train the model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
        
        print("Training model...")
        rf_model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"Model performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f} (correct buy signals)")
        print(f"Recall: {recall:.4f} (captured opportunities)")
        
        # Store the model
        self.model = rf_model
        return self.model
    
    def save_model(self, path="ai_stock_screener_model.pkl"):
        """Save the trained model"""
        if self.model:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {path}")
        else:
            print("No model to save. Train a model first.")
    
    def load_model(self, path="ai_stock_screener_model.pkl"):
        """Load a trained model"""
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_signals(self):
        """Generate buy/sell signals for all stocks"""
        if not self.model:
            print("No model available. Train or load a model first.")
            return None
        
        signals = {}
        feature_columns = [col for col in next(iter(self.data.values())).columns 
                          if col not in ['Target', 'Future_Return']]
        
        for symbol, df in self.data.items():
            if len(df) > 0:
                # Get the most recent data
                latest_data = df.iloc[-1:][feature_columns].copy()
                latest_data['Symbol'] = self.stocks.index(symbol) if symbol in self.stocks else -1
                
                # Make prediction
                prediction = self.model.predict(latest_data)[0]
                probability = self.model.predict_proba(latest_data)[0][1]
                
                # Store signal
                signals[symbol] = {
                    'Signal': 'BUY' if prediction == 1 else 'HOLD',
                    'Confidence': probability,
                    'Close': df['Close'].iloc[-1],
                    'RSI': df['RSI'].iloc[-1],
                    'MACD': df['MACD'].iloc[-1],
                    'MACD_Signal': df['MACD_Signal'].iloc[-1],
                    'BB_%B': df['BB_%B'].iloc[-1],
                    'Stoch_K': df['Stoch_K'].iloc[-1],
                    'Stoch_D': df['Stoch_D'].iloc[-1],
                    'SMA_50': df['SMA_50'].iloc[-1],
                    'SMA_200': df['SMA_200'].iloc[-1]
                }
        
        # Convert to DataFrame for easier viewing
        signals_df = pd.DataFrame.from_dict(signals, orient='index')
        signals_df.sort_values(by='Confidence', ascending=False, inplace=True)
        
        return signals_df
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        self.fetch_data()
        self.generate_features()
        self.train_model()
        return self.generate_signals()

# Example usage
if __name__ == "__main__":
    # Initialize the screener
    screener = AIStockScreener()
    
    # Run complete analysis
    signals = screener.run_complete_analysis()
    
    # Display top buy signals
    buy_signals = signals[signals['Signal'] == 'BUY']
    print("\nTop Buy Signals:")
    print(buy_signals.head(10))
    
    # Save the model for future use
    screener.save_model()