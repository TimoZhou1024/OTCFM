"""
OT-CFM: Optimal Transport Coupled Flow Matching for Multi-View Clustering
"""

from .ot_cfm import OTCFM
from .config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    ExperimentConfig,
    get_default_config,
)
from .datasets import (
    MultiViewDataset,
    get_dataset,
    get_dataloader,
    create_dataloader,
)
from .trainer import Trainer
from .metrics import evaluate_clustering, MetricTracker
from .losses import OTCFMLoss

__version__ = "0.1.0"

__all__ = [
    "OTCFM",
    "ModelConfig",
    "TrainingConfig", 
    "DataConfig",
    "ExperimentConfig",
    "get_default_config",
    "MultiViewDataset",
    "get_dataset",
    "get_dataloader",
    "create_dataloader",
    "Trainer",
    "evaluate_clustering",
    "MetricTracker",
    "OTCFMLoss",
]
