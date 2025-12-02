"""
Configuration file for OT-CFM Multi-View Clustering
"""

import argparse
import torch
from dataclasses import dataclass, field
from typing import List, Optional


def get_device() -> str:
    """Get the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    latent_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    num_clusters: int = 10
    sigma_min: float = 1e-4
    
    # Flow matching
    flow_num_layers: int = 4
    flow_hidden_dim: int = 256
    time_dim: int = 64
    ode_steps: int = 10  # Number of Euler steps
    
    # Kernel for GW loss
    kernel_type: str = "rbf"  # "rbf", "cosine", "linear"
    kernel_gamma: float = 1.0
    
    # Loss weights
    lambda_gw: float = 0.1
    lambda_cluster: float = 0.5
    lambda_recon: float = 1.0
    lambda_contrastive: float = 0.1
    
    # Dropout
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 200
    warmup_epochs: int = 10
    
    # Optimizer
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine", "step", "none"
    
    # Alternating optimization
    cluster_update_freq: int = 5  # Update clustering every N epochs
    
    # Gradient clipping
    clip_grad_norm: float = 1.0
    
    # Early stopping
    patience: int = 20
    
    # Random seed
    seed: int = 42
    
    # Device (auto-detect)
    device: str = field(default_factory=get_device)


@dataclass  
class DataConfig:
    """Dataset configuration"""
    dataset_name: str = "Caltech101"  # Dataset name
    data_root: str = "./data"
    missing_rate: float = 0.0  # Rate of missing views
    unaligned_rate: float = 0.0  # Rate of unaligned samples
    num_views: int = 2


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment settings
    experiment_name: str = "ot_cfm"
    save_dir: str = "./results"
    log_interval: int = 10
    save_model: bool = True
    
    # Ablation settings
    ablation_mode: str = "full"  # "full", "no_gw", "no_cluster", "no_ot", "no_flow"
    
    # Device (kept for backward compatibility, auto-detect)
    device: str = field(default_factory=get_device)
    num_workers: int = 4


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration"""
    return ExperimentConfig()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="OT-CFM Multi-View Clustering")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="Caltech101",
                       choices=["Caltech101", "Scene15", "NoisyMNIST", 
                               "BDGP", "CUB", "NUS-WIDE", "Reuters", "Synthetic"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--missing_rate", type=float, default=0.0)
    parser.add_argument("--unaligned_rate", type=float, default=0.0)
    
    # Model
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--ode_steps", type=int, default=10)
    parser.add_argument("--flow_num_layers", type=int, default=4)
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    
    # Loss weights
    parser.add_argument("--lambda_gw", type=float, default=0.1)
    parser.add_argument("--lambda_cluster", type=float, default=0.5)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_contrastive", type=float, default=0.1)
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--optimizer", type=str, default="adamw", 
                       choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "step", "none"])
    
    # Experiment
    parser.add_argument("--experiment_name", type=str, default="ot_cfm")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Ablation
    parser.add_argument("--ablation_mode", type=str, default="full",
                       choices=["full", "no_gw", "no_cluster", "no_ot", "no_flow"])
    
    # Comparison
    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--run_ablation", action="store_true")
    
    args = parser.parse_args()
    return args


def args_to_config(args: argparse.Namespace) -> ExperimentConfig:
    """Convert parsed args to config object"""
    model_config = ModelConfig(
        latent_dim=args.latent_dim,
        num_clusters=args.num_clusters,
        ode_steps=args.ode_steps,
        flow_num_layers=args.flow_num_layers,
        flow_hidden_dim=args.flow_hidden_dim,
        lambda_gw=args.lambda_gw,
        lambda_cluster=args.lambda_cluster,
        lambda_recon=args.lambda_recon,
        lambda_contrastive=args.lambda_contrastive,
    )
    
    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        seed=args.seed,
        device=args.device,
    )
    
    data_config = DataConfig(
        dataset_name=args.dataset,
        data_root=args.data_root,
        missing_rate=args.missing_rate,
        unaligned_rate=args.unaligned_rate,
    )
    
    config = ExperimentConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        experiment_name=args.experiment_name,
        save_dir=args.save_dir,
        ablation_mode=args.ablation_mode,
        device=args.device,
    )
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print(f"Model config: {config.model}")
    print(f"Training config: {config.training}")
    print(f"Data config: {config.data}")


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="OT-CFM Multi-View Clustering")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="Caltech101",
                       choices=["Caltech101", "Scene15", "NoisyMNIST", 
                               "BDGP", "CUB", "NUS-WIDE", "Reuters"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--missing_rate", type=float, default=0.0)
    parser.add_argument("--unaligned_rate", type=float, default=0.0)
    
    # Model
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--ode_steps", type=int, default=10)
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lambda_gw", type=float, default=0.1)
    parser.add_argument("--lambda_cluster", type=float, default=0.5)
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default="ot_cfm")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Ablation
    parser.add_argument("--ablation_mode", type=str, default="full",
                       choices=["full", "no_gw", "no_cluster", "no_ot", "no_flow"])
    
    # Comparison
    parser.add_argument("--run_baselines", action="store_true")
    parser.add_argument("--run_ablation", action="store_true")
    
    args = parser.parse_args()
    return args


def args_to_config(args) -> ExperimentConfig:
    """Convert parsed args to config object"""
    model_config = ModelConfig(
        latent_dim=args.latent_dim,
        num_clusters=args.num_clusters,
    )
    
    training_config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lambda_gw=args.lambda_gw,
        lambda_cluster=args.lambda_cluster,
        ode_steps=args.ode_steps,
        seed=args.seed,
    )
    
    data_config = DataConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        missing_rate=args.missing_rate,
        unaligned_rate=args.unaligned_rate,
    )
    
    config = ExperimentConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        exp_name=args.exp_name,
        save_dir=args.save_dir,
        ablation_mode=args.ablation_mode,
        device=args.device,
    )
    
    return config
