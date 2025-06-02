"""Main script for dispatching XGBoost training on CPU, GPU, or Dask."""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from configs import get_config
from libs import xgb_trainer, xgb_dask_trainer


def get_trainer(label):
    """Select the appropriate trainer module based on label."""
    if label == "DASK":
        return xgb_dask_trainer
    return xgb_trainer

def train_model(label, label_param_map, X_train, X_test, y_train, y_test, config):
    trainer = get_trainer(label)
    params = label_param_map[label]
    trainer.train_model(X_train, X_test, y_train, y_test,
                        params=params, num_boost_round=config.common.num_boost_round, label=label)

def main():
    """Run XGBoost training on selected backends."""
    label = None
    
    config = get_config()

    X, y = make_classification(n_samples=1_000_000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    label_param_map = {
        "CPU": {**config.common, **config.cpu},
        "GPU": {**config.common, **config.gpu},
        "DASK": {**config.common},
    }

    if label:
        train_model(label, label_param_map, X_train, X_test, y_train, y_test, config)
        return
    
    for label in label_param_map:
        train_model(label, label_param_map, X_train, X_test, y_train, y_test, config)


if __name__ == "__main__":
    main()
