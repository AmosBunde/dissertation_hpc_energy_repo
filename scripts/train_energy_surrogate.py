#!/usr/bin/env python3
import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)  # e.g., energy_j
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--out", default=None)      # optional custom output path
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2  = r2_score(y_test, pred)
    print(f"[metrics] MAE={mae:.4f}  R2={r2:.4f}  n={len(y_test)}")

    os.makedirs("models", exist_ok=True)
    out_path = args.out or f"models/{args.target.replace('_','-')}_surrogate_rf.joblib"
    joblib.dump(model, out_path)
    print(f"[OK] Saved model -> {out_path}")

if __name__ == "__main__":
    main()
