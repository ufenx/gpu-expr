"""XGBoost training configuration using ml_collections."""

from ml_collections import ConfigDict


def get_config():
    """Returns training configuration."""
    config = ConfigDict()

    config.common = ConfigDict()
    config.common.objective = 'binary:logistic'
    config.common.verbosity = 0
    config.common.eta = 0.3
    config.common.tree_method = 'hist'
    config.common.num_boost_round = 100

    config.cpu = ConfigDict()
    config.cpu.device = None  # Use default CPU

    config.gpu = ConfigDict()
    config.gpu.device = 'cuda'

    config.dask = ConfigDict()
    config.dask.device = 'cuda'

    return config
