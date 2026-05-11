"""
prepare_data.py
---------------
Cleans and prepares the raw housing data for model training.
Run this script to reproduce the exact same processed dataset.

Usage:
    python src/prepare_data.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

# ── CONFIG ────────────────────────────────────────────────────
RAW_DATA_PATH      = "data/raw/housing.csv"
PROCESSED_DIR      = "data/processed"
TRAIN_PATH         = "data/processed/train.csv"
TEST_PATH          = "data/processed/test.csv"
STATS_PATH         = "data/processed/data_stats.json"
RANDOM_STATE       = 42
TEST_SIZE          = 0.2

def load_data():
    print("📂 Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df

def explore_data(df):
    print("\n🔍 Exploring data...")
    print(f"   Missing values:\n{df.isnull().sum()}")
    print(f"\n   Basic stats:")
    print(df.describe().round(2))
    return df

def clean_data(df):
    print("\n🧹 Cleaning data...")
    before = len(df)

    # Fill missing values with median
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   Filled {col} missing values with median: {median_val:.2f}")

    # Remove extreme outliers (beyond 3 standard deviations)
    for col in ['MedHouseVal', 'MedInc', 'AveRooms']:
        mean = df[col].mean()
        std  = df[col].std()
        before_col = len(df)
        df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
        removed = before_col - len(df)
        if removed > 0:
            print(f"   Removed {removed} outliers from {col}")

    print(f"   Rows before: {before} → after: {len(df)}")
    return df

def engineer_features(df):
    print("\n⚙️  Engineering features...")

    # Rooms per household
    df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']

    # Bedrooms ratio
    df['BedroomsRatio'] = df['AveBedrms'] / df['AveRooms']

    # Population per household
    df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']

    # Distance from city center (Los Angeles: 34.05, -118.24)
    df['DistFromCenter'] = np.sqrt(
        (df['Latitude']  - 34.05) ** 2 +
        (df['Longitude'] - (-118.24)) ** 2
    )

    print(f"   New features added: RoomsPerHousehold, BedroomsRatio, PopulationPerHousehold, DistFromCenter")
    print(f"   Total features now: {df.shape[1] - 1} (excluding target)")
    return df

def scale_features(df):
    print("\n📏 Scaling features...")

    target_col = 'MedHouseVal'
    feature_cols = [c for c in df.columns if c != target_col]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f"   Scaled {len(feature_cols)} features using StandardScaler")
    return df

def split_data(df):
    print("\n✂️  Splitting data...")

    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_df = X_train.copy()
    train_df['MedHouseVal'] = y_train

    test_df = X_test.copy()
    test_df['MedHouseVal'] = y_test

    print(f"   Train set: {len(train_df)} rows")
    print(f"   Test set:  {len(test_df)} rows")
    return train_df, test_df

def save_data(train_df, test_df):
    print("\n💾 Saving processed data...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH,  index=False)

    # Save stats for MLflow logging later
    stats = {
        "train_rows":    len(train_df),
        "test_rows":     len(test_df),
        "features":      [c for c in train_df.columns if c != 'MedHouseVal'],
        "target":        "MedHouseVal",
        "test_size":     TEST_SIZE,
        "random_state":  RANDOM_STATE
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"   ✅ Train data → {TRAIN_PATH}")
    print(f"   ✅ Test data  → {TEST_PATH}")
    print(f"   ✅ Stats      → {STATS_PATH}")

def main():
    print("=" * 50)
    print("  DATA PREPARATION PIPELINE")
    print("=" * 50)

    df = load_data()
    df = explore_data(df)
    df = clean_data(df)
    df = engineer_features(df)
    df = scale_features(df)
    train_df, test_df = split_data(df)
    save_data(train_df, test_df)

    print("\n" + "=" * 50)
    print("  ✅ DATA PREPARATION COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
