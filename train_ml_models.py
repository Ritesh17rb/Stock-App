#!/usr/bin/env python3
"""
Script to train ML models on your NIFTY-50 data
"""

import os
import sys
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_models.model_trainer import StockModelTrainer


def check_data_files(data_path='stock_data'):
    """Check if data files exist and are readable"""
    print("🔍 Checking data files...")

    if not os.path.exists(data_path):
        print(f"❌ Error: Directory '{data_path}' not found!")
        print("Please create a 'stock_data' directory and add your CSV files")
        return []

    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

    if not csv_files:
        print(f"❌ No CSV files found in '{data_path}'")
        print("Please add your NIFTY-50 CSV files to the stock_data directory")
        return []

    # Check first file to understand structure
    if csv_files:
        first_file = os.path.join(data_path, csv_files[0])
        try:
            sample_df = pd.read_csv(first_file, nrows=5)
            print(f"📊 Sample data from {csv_files[0]}:")
            print(f"   Columns: {list(sample_df.columns)}")
            print(f"   First row: {dict(sample_df.iloc[0])}")
        except Exception as e:
            print(f"❌ Error reading {csv_files[0]}: {e}")

    symbols = [f.replace('.csv', '') for f in csv_files]
    print(f"✅ Found {len(symbols)} stock files")
    return symbols


def main():
    print("🚀 Starting ML Model Training for NIFTY-50 Stocks...")
    print("=" * 60)

    # Check data files
    symbols = check_data_files('stock_data')

    if not symbols:
        print("\n💡 Please ensure:")
        print("   1. You have a 'stock_data' directory")
        print("   2. It contains CSV files for NIFTY-50 stocks")
        print("   3. Each CSV has columns like Date, Open, High, Low, Close, Volume")
        return

    print(f"\n📈 Symbols to process: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

    # Initialize trainer
    trainer = StockModelTrainer()

    print("\n🎯 Starting model training...")
    print("=" * 60)

    # Train models
    results = trainer.train_all_models(symbols)

    # Analyze results
    successful = [s for s, r in results.items() if r['status'] == 'success']
    failed = [s for s, r in results.items() if r['status'] == 'failed']

    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {len(successful)} stocks")
    print(f"❌ Failed: {len(failed)} stocks")

    if successful:
        print(f"\n🎉 Successfully trained models for:")
        for symbol in successful[:10]:
            mae = results[symbol].get('mae', 'N/A')
            if isinstance(mae, (int, float)):
                print(f"   • {symbol} (MAE: {mae:.4f})")
            else:
                print(f"   • {symbol}")

        if len(successful) > 10:
            print(f"   ... and {len(successful) - 10} more")

    if failed:
        print(f"\n⚠️ Failed to train models for:")
        for symbol in failed[:10]:
            reason = results[symbol].get('reason', 'Unknown error')
            print(f"   • {symbol} - {reason}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")

    # Calculate success rate
    success_rate = (len(successful) / len(symbols)) * 100
    print(f"\n📈 Success rate: {success_rate:.1f}%")

    if successful:
        print("\n🎊 Training completed! You can now run the Flask application.")
        print("👉 Run: python app.py")
    else:
        print("\n😞 No models were successfully trained.")
        print("💡 Check your data format and try again.")


if __name__ == "__main__":
    main()