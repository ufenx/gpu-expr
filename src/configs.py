"""XGBoost training configuration using ml_collections."""

from ml_collections import ConfigDict


def get_config():
    """Returns training configuration."""
    config = ConfigDict()

    config.sample = ConfigDict()
    config.sample.n_samples = 1_000_000            # For nontesting, change to 100_000_000
    config.n_features = 20
    config.random_state = 42
    config.test_size = 0.2
    
    config.common = ConfigDict()
    config.common.booster = "gbtree"
    config.common.objective = 'binary:logistic'
    config.common.eta = 0.3
    config.common.gemma = 0
    config.common.max_depth = 6
    config.common.min_child_weight = 1
    config.common.max_delta_step = 0
    config.common.subsample = 1
    config.common.sampling_method = "uniform"
    config.common.colsample_bytree = 1
    config.common.colsample_bylevel = 1
    config.common.colsample_bynode = 1
    config.common.lmbda = 1
    config.common.alpha = 0
    config.common.tree_method = 'hist'
    config.common.scale_pos_weight = 1
    config.common.grow_policy = "depthwise"
    config.common.max_leaves = 0
    config.common.max_bin = 256
    config.common.num_boost_round = 100

    config.common.verbosity = 0

    config.cpu = ConfigDict()
    config.cpu.device = None  # Use default CPU

    config.gpu = ConfigDict()
    config.gpu.device = 'cuda'

    config.dask = ConfigDict()
    config.dask.device = 'cuda'

    return config
