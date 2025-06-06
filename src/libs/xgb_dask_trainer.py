"""XGBoost Dask training logic."""

import time
from xgboost import dask as dxgb
import dask.array as da
import dask.distributed
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

import os
import sys
sys.stderr = open(os.devnull, 'w')

import dask
dask.config.set({'logging.distributed': 'error'})

def train_model(X_train=None, X_test=None, y_train=None, y_test=None,
                params=None, num_boost_round=100, label="DASK", log_num=0):
    """Train XGBoost model using Dask (distributed)."""
    print(f"Training with {label}...")

    cluster = dask.distributed.LocalCluster()
    client = dask.distributed.Client(cluster)

    X_train_da = da.from_array(X_train, chunks=(10000, -1))
    X_test_da = da.from_array(X_test, chunks=(10000, -1))
    y_train_da = da.from_array(y_train, chunks=(10000,))
    y_test_da = da.from_array(y_test, chunks=(10000,))

    dtrain = dxgb.DaskDMatrix(client, X_train_da, y_train_da)
    dtest = dxgb.DaskDMatrix(client, X_test_da, y_test_da)

    start = time.time()
    model = dxgb.train(client, params, dtrain, num_boost_round=num_boost_round)
    end = time.time()

    y_pred_prob = dxgb.predict(client, model, dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    y_true = y_test_da.compute()
    y_pred = y_pred.compute()
    acc = accuracy_score(y_true, y_pred)

    print(f"{label} Accuracy: {acc:.4f}")
    print(f"{label} Training Time: {end - start:.2f} seconds\n")

    os.makedirs("log", exist_ok=True)
    log_path = f"log/{label}_{log_num}.log"
    with open(log_path, "") as f:
        f.write(f"{acc:.2f}, {elapsed:.6f}\n")

    client.close()
    cluster.close()
