import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


class StockPredictor:
    def __init__(self, models_dir='ml_models/trained_models'):
        self.models_dir = models_dir
        self.loaded_models = {}

    def load_model(self, symbol):
        """Load trained model for symbol"""
        if symbol in self.loaded_models:
            return self.loaded_models[symbol]

        model_path = os.path.join(self.models_dir, f'{symbol}_model.joblib')
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                self.loaded_models[symbol] = model_data
                return model_data
            except Exception as e:
                print(f"Error loading model for {symbol}: {e}")
                return None
        return None

    def get_available_models(self):
        """Get list of available trained models"""
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_model.joblib')]
        return [f.replace('_model.joblib', '') for f in model_files]

    def predict_future_price(self, symbol, current_data):
        """Predict next day price using ML model"""
        model_data = self.load_model(symbol)
        if not model_data:
            return None

        try:
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']

            # Prepare features for prediction
            features = current_data[feature_columns].values.reshape(1, -1)
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]
            return float(prediction)

        except Exception as e:
            print(f"Prediction error for {symbol}: {e}")
            return None

    def get_price_trend(self, symbol, days=30):
        """Get price trend analysis"""
        # This would integrate with your existing data
        # For now, returning mock analysis
        trends = {
            'short_term': 'bullish',
            'medium_term': 'neutral',
            'support_level': 1500.50,
            'resistance_level': 1650.75,
            'volatility': 'medium'
        }
        return trends