import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

def fetch_data():
    print("📦 Downloading California Housing dataset...")
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/housing.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset saved to {output_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"\n🔍 First 3 rows:")
    print(df.head(3))

if __name__ == "__main__":
    fetch_data()
