"""XGBoost CPU/GPU training logic."""

import time
import xgboost as xgb
from sklearn.metrics import accuracy_score


def train_model(X_train, X_test, y_train, y_test, params, num_boost_round=100, label="CPU"):
    """Train XGBoost model using CPU or GPU backend."""
    print(f"Training with {label}...")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    start_time = time.time()
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    elapsed = time.time() - start_time

    y_pred = model.predict(dtest)
    y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]

    acc = accuracy_score(y_test, y_pred_binary)
    print(f"{label} Accuracy: {acc:.4f}")
    print(f"{label} Training Time: {elapsed:.2f} seconds\n")
