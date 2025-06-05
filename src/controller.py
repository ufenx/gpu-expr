"""Main script for dispatching XGBoost training on CPU, GPU, or Dask."""

import argparse
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from configs import get_config

import os
import subprocess

def get_available_gpus(label):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, text=True
        )
        gpu_ids = result.stdout.strip().split('\n')
        if label == "CUDA":
            return [f"cuda:{gpu_id.strip()}" for gpu_id in gpu_ids]
        else:
            return [f"gpu:{gpu_id.strip()}" for gpu_id in gpu_ids]
    except (subprocess.SubprocessError, FileNotFoundError):
        print("[WARN] Could not detect GPUs automatically. Falling back to manual input.")
        while True:
            try:
                count = int(input("Enter the number of GPUs you want to use: "))
                if label == "CUDA":
                    return [f"cuda:{gpu_id.strip()}" for gpu_id in gpu_ids]
                else:
                    return [f"gpu:{gpu_id.strip()}" for gpu_id in gpu_ids]
            except ValueError:
                print("Invalid input. Please enter an integer.")

def get_trainer(label):
    """Select the appropriate trainer module based on label."""
    try:
        if label == "DASK":
            from libs import xgb_dask_trainer
            return xgb_dask_trainer
        elif label == "JAX":
            from libs import xgb_jax_trainer
            return xgb_jax_trainer
        elif label == "JIT":
            from libs import xgb_jit_trainer
            return xgb_jit_trainer
        else:
            from libs import xgb_trainer
            return xgb_trainer
    except ModuleNotFoundError as e:
        missing_module = str(e).split("'")[1]
        print(f"[ERROR] Module '{missing_module}' not found in the current environment. Please switch to the correct Conda environment.")
        raise

def train_model(label, X_train, X_test, y_train, y_test, config):
    """Train model using the selected trainer and parameters."""
    trainer = get_trainer(label)
    label_param_map = {
        "CPU":  {**config.common, **config.cpu},
        "GPU":  {**config.common, **config.gpu},
        "CUDA": {**config.common, **config.cuda},
        "JAX":  {**config.common, **config.gpu},
        "JIT":  {**config.common, **config.gpu},
        "DASK": {**config.common},
    }
    params = label_param_map[label]
    trainer.train_model(X_train, X_test, y_train, y_test,
                        params=params, num_boost_round=config.common.num_boost_round, label=label)

def generate_data(config):
    """Generate and split classification dataset once."""
    X, y = make_classification(n_samples=config.sample.n_samples,
                               n_features=config.sample.n_features, 
                               random_state=config.sample.random_state)
    return train_test_split(X, y, test_size=config.sample.test_size,
                            random_state=config.sample.random_state)

def select_label(label_param_list):
    """Interactively select label if not provided."""
    print("Select a backend to run the XGBoost training:")
    label_options = label_param_list
    for idx, option in enumerate(label_options):
        print(f"{idx + 1}. {option}")
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(label_options):
                return label_options[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(label_options)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run XGBoost training on selected backend.")
    parser.add_argument('-n', type=int, default=1, help="Number of executions to run")
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_config()
    
    label_param_list = ["CPU", "GPU", "CUDA", "JAX", "JIT", "DASK"]
    label = select_label(label_param_list)

    gpu_ids = get_available_gpus(label)

    # Generate dataset once
    X_train, X_test, y_train, y_test = generate_data(config)

    for i in range(args.n):
        print(f"\n--- Training Run #{i + 1} ---")
        train_model(label, X_train, X_test, y_train, y_test, config)

if __name__ == "__main__":
    main()
