import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class StockModelTrainer:
    def __init__(self, models_dir='ml_models/trained_models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def load_stock_data(self, symbol, data_path='stock_data'):
        """Load NSE India format stock data"""
        file_path = os.path.join(data_path, f'{symbol}.csv')
        try:
            # Read CSV with proper parsing
            df = pd.read_csv(file_path)

            # Convert Date column
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')

            # Select only the columns we need for ML
            # Using: Date, Open, High, Low, Close, Volume
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

            # Check if all required columns exist
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  ⚠️ Missing columns: {missing_cols}")
                return None

            # Create a clean dataframe with only required columns
            clean_df = df[required_cols].copy()

            # Convert to numeric (should already be numeric based on your check)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

            # Drop any rows with NaN in essential columns
            clean_df = clean_df.dropna()

            if len(clean_df) < 100:
                print(f"  ⚠️ Insufficient data: {len(clean_df)} rows after cleaning")
                return None

            print(f"  ✅ Loaded {symbol}: {len(clean_df)} records")
            return clean_df

        except Exception as e:
            print(f"  ❌ Error loading {symbol}: {e}")
            return None

    def create_features(self, df):
        """Create technical indicators from NSE price data"""
        data = df.copy()

        # Basic price features
        data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100
        data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100
        data['Price_Range'] = (data['High'] - data['Low']) / data['Close']

        # Moving averages
        windows = [5, 10, 20]
        for window in windows:
            if len(data) >= window:
                data[f'MA_{window}'] = data['Close'].rolling(window=window, min_periods=1).mean()
                data[f'Volume_MA_{window}'] = data['Volume'].rolling(window=window, min_periods=1).mean()

        # Volatility measures
        if len(data) >= 10:
            data['Volatility_10'] = data['Close'].rolling(10, min_periods=5).std()
            data['Volatility_20'] = data['Close'].rolling(20, min_periods=10).std()

        # RSI
        data['RSI_14'] = self.calculate_rsi(data['Close'])

        # MACD
        data['MACD'] = self.calculate_macd(data['Close'])

        # Bollinger Bands
        data['BB_Upper'], data['BB_Lower'] = self.calculate_bollinger_bands(data['Close'])
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

        # Price momentum and lag features
        for lag in [1, 2, 3, 5]:
            data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_lag_{lag}'] = data['Volume'].shift(lag)
            data[f'Return_lag_{lag}'] = data['Close'].pct_change(lag)

        # Volume features
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_5']

        # Target: Next day closing price
        data['Target'] = data['Close'].shift(-1)

        # Drop rows with NaN values (from rolling windows and shifts)
        data = data.dropna()

        if len(data) < 50:
            return None

        return data

    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, slow=26, fast=12):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window, min_periods=1).mean()
        rolling_std = prices.rolling(window=window, min_periods=1).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def prepare_features_for_training(self, feature_df):
        """Prepare features for model training"""
        # Select feature columns (exclude Date, Target, and Close)
        exclude_cols = ['Date', 'Target', 'Close']
        feature_columns = [col for col in feature_df.columns
                           if col not in exclude_cols and
                           not col.startswith('Date')]

        # Ensure all features are numeric
        X = feature_df[feature_columns].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Remove any columns that are entirely NaN
        X = X.dropna(axis=1, how='all')

        # Update feature columns list
        feature_columns = [col for col in feature_columns if col in X.columns]

        if not feature_columns:
            return None, None, None

        y = feature_df['Target']

        # Align X and y indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        y = y.loc[X.index]

        return X, y, feature_columns

    def train_model_for_stock(self, symbol):
        """Train ML model for a specific stock"""
        print(f"    Processing {symbol}...")

        # Load data
        df = self.load_stock_data(symbol)
        if df is None:
            return None, None

        # Create features
        feature_df = self.create_features(df)
        if feature_df is None or len(feature_df) < 100:
            print(f"    ⚠️ Insufficient features for {symbol}: {len(feature_df) if feature_df else 0} samples")
            return None, None

        print(f"    📊 Generated {len(feature_df.columns)} features")

        # Prepare features for training
        X, y, feature_columns = self.prepare_features_for_training(feature_df)
        if X is None or len(X) < 100:
            print(f"    ⚠️ Insufficient data after feature prep: {len(X) if X else 0} samples")
            return None, None

        print(f"    🎯 Using {len(feature_columns)} features for training")

        # Split data (chronological split - important for time series)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if len(X_train) < 50 or len(X_test) < 25:
            print(f"    ⚠️ Insufficient train/test split: {len(X_train)} train, {len(X_test)} test")
            return None, None

        try:
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Initialize models (simpler models for better convergence)
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=10
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=6,
                    learning_rate=0.1
                ),
                'linear_regression': LinearRegression()
            }

            best_model = None
            best_score = float('inf')
            best_model_name = None

            for name, model in models.items():
                try:
                    print(f"      Training {name}...")
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                    if mae < best_score:
                        best_score = mae
                        best_model = model
                        best_model_name = name

                    print(f"      {name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")

                except Exception as e:
                    print(f"      ❌ Error training {name}: {e}")
                    continue

            if best_model:
                # Save model and scaler
                model_data = {
                    'model': best_model,
                    'scaler': scaler,
                    'feature_columns': feature_columns,
                    'performance': {
                        'MAE': best_score,
                        'data_points': len(X),
                        'train_samples': len(X_train),
                        'test_samples': len(X_test)
                    },
                    'model_name': best_model_name,
                    'last_training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
                }

                model_path = os.path.join(self.models_dir, f'{symbol}_model.joblib')
                joblib.dump(model_data, model_path)

                print(f"    ✅ {symbol}: {best_model_name} (MAE: {best_score:.2f})")
                return best_model_name, best_score
            else:
                print(f"    ❌ No model trained for {symbol}")
                return None, None

        except Exception as e:
            print(f"    ❌ Training failed for {symbol}: {e}")
            return None, None

    def train_all_models(self, symbols):
        """Train models for all available symbols"""
        results = {}
        total = len(symbols)

        successful_count = 0

        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{total}] Training model for {symbol}...")
            model_name, score = self.train_model_for_stock(symbol)

            if model_name:
                results[symbol] = {
                    'model': model_name,
                    'mae': score,
                    'status': 'success',
                    'data_points': 'available'
                }
                successful_count += 1
            else:
                results[symbol] = {
                    'status': 'failed',
                    'reason': 'Insufficient data or training error'
                }

            print(f"    Progress: {successful_count}/{i} successful")
            print("-" * 50)

        # Save training summary
        summary_path = os.path.join(self.models_dir, 'training_summary.joblib')
        joblib.dump(results, summary_path)

        print(f"\n🎯 Final Results: {successful_count}/{total} stocks trained successfully")
        return results

    def get_training_summary(self):
        """Get summary of trained models"""
        summary_path = os.path.join(self.models_dir, 'training_summary.joblib')
        if os.path.exists(summary_path):
            return joblib.load(summary_path)
        return {}