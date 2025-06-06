"""XGBoost trainer with custom objective computed using JAX."""

import time
import xgboost as xgb
import jax.numpy as jnp
import numpy as np
from sklearn.metrics import accuracy_score
import os


def custom_obj_jax(preds, dtrain):
    """Binary logistic loss gradient/hessian using JAX."""
    labels = jnp.array(dtrain.get_label())
    preds = jnp.array(preds)

    preds_sigmoid = 1 / (1 + jnp.exp(-preds))
    grad_val = preds_sigmoid - labels
    hess_val = preds_sigmoid * (1 - preds_sigmoid)

    return np.array(grad_val), np.array(hess_val)


def train_model(X_train, X_test, y_train, y_test, params, num_boost_round=100, label="JAX", log_num=0):
    """Train XGBoost model with JAX-based custom objective."""
    print(f"Training with {label}...")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    start_time = time.time()
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, obj=custom_obj_jax)
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
