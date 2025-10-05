import pandas as pd
import os


def check_stock_data_files():
    """Check the format of stock data files"""
    data_path = 'stock_data'

    if not os.path.exists(data_path):
        print("❌ 'stock_data' directory not found!")
        return

    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

    print(f"Found {len(csv_files)} CSV files")
    print("\nChecking first 3 files...")

    for i, file in enumerate(csv_files[:3]):
        print(f"\n{'=' * 50}")
        print(f"File: {file}")
        print(f"{'=' * 50}")

        try:
            df = pd.read_csv(os.path.join(data_path, file))
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"First 2 rows:")
            for col in df.columns:
                print(f"  {col}: {df[col].iloc[0]} (type: {type(df[col].iloc[0])})")

            # Check for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            print(f"\nNumeric columns: {list(numeric_cols)}")

        except Exception as e:
            print(f"Error reading file: {e}")


if __name__ == "__main__":
    check_stock_data_files()