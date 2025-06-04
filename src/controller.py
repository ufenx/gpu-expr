"""Main script for dispatching XGBoost training on CPU, GPU, or Dask."""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from configs import get_config

def get_trainer(label):
    """Select the appropriate trainer module based on label."""
    try:
        if label == "DASK":
            from libs import xgb_dask_trainer
            return xgb_dask_trainer
        elif label == "JAX":
            from libs import xgb_jax_trainer
            return xgb_jax_trainer
        else:
            from libs import xgb_trainer
            return xgb_trainer
    except ModuleNotFoundError as e:
        missing_module = str(e).split("'")[1]
        print(f"[ERROR] Module '{missing_module}' not found in the current environment. Please switch to the correct Conda environment.")
        raise

def train_model(label, label_param_map, X_train, X_test, y_train, y_test, config):
    trainer = get_trainer(label)
    params = label_param_map[label]
    trainer.train_model(X_train, X_test, y_train, y_test,
                        params=params, num_boost_round=config.common.num_boost_round, label=label)

def main():
    """Run XGBoost training on selected backends."""
    config = get_config()
    label = config.label

    X, y = make_classification(n_samples=config.sample.n_samples, n_features=config.sample.n_features, 
                               random_state=config.sample.random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.sample.test_size,
                                                        random_state=config.sample.random_state)

    label_param_map = {
        "CPU": {**config.common, **config.cpu},
        "GPU": {**config.common, **config.gpu},
        "CUDA":  {**config.common, **config.cuda},
        "JAX": {**config.common, **config.gpu},
        "DASK": {**config.common},
    }

    if not label:
        print("Select a backend to run the XGBoost training:")
        label_options = list(label_param_map.keys())
        for idx, option in enumerate(label_options):
            print(f"{idx + 1}. {option}")
        
        while True:
            try:
                choice = int(input("Enter the number of your choice: "))
                if 1 <= choice <= len(label_options):
                    label = label_options[choice - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(label_options)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
    train_model(label, label_param_map, X_train, X_test, y_train, y_test, config)

    # Remove the one click solution as different environments may not be compatible.
    # for label in label_param_map:
    #     train_model(label, label_param_map, X_train, X_test, y_train, y_test, config)


if __name__ == "__main__":
    main()
