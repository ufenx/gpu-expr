"""XGBoost trainer with custom objective computed using PyTorch."""

import time
import xgboost as xgb
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import os


def custom_obj_torch(preds, dtrain):
    """Binary logistic loss gradient/hessian using PyTorch."""
    labels = torch.tensor(dtrain.get_label(), dtype=torch.float32)
    preds = torch.tensor(preds, dtype=torch.float32)

    preds_sigmoid = torch.sigmoid(preds)
    grad_val = preds_sigmoid - labels
    hess_val = preds_sigmoid * (1.0 - preds_sigmoid)

    return grad_val.numpy(), hess_val.numpy()


def train_model(X_train, X_test, y_train, y_test, params, num_boost_round=100, label="Torch", log_num=0):
    """Train XGBoost model with PyTorch-based custom objective."""
    print(f"Training with {label}...")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    start_time = time.time()
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, obj=custom_obj_torch)
    elapsed = time.time() - start_time

    preds = model.predict(dtest)
    preds_binary = [1 if p > 0.5 else 0 for p in preds]
    acc = accuracy_score(y_test, preds_binary)

    print(f"{label} Accuracy: {acc:.4f}")
    print(f"{label} Training Time: {elapsed:.2f} seconds\n")

    os.makedirs("log", exist_ok=True)
    log_path = f"log/{label}_{log_num}.log"
    with open(log_path, "a") as f:
        f.write(f"{acc:.2f}, {elapsed:.6f}\n")
