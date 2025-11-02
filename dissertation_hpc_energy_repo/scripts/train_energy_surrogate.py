#!/usr/bin/env python3
"""
train_energy_surrogate.py - simple regression from job features -> energy (J).

Usage:
  python train_energy_surrogate.py --csv training.csv --target energy_j
"""
import argparse, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="training CSV")
    ap.add_argument("--target", default="energy_j", help="target column")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found in CSV.")

    y = df[args.target]
    X = df.drop(columns=[args.target])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Test MAPE: {mape:.3f}")
    joblib.dump(model, "energy_surrogate_rf.joblib")
    print("Saved model -> energy_surrogate_rf.joblib")

if __name__ == "__main__":
    main()
