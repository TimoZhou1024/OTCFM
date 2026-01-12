"""
Optuna hyperparameter tuning for OT-CFM
Automatically searches for optimal hyperparameters using Bayesian optimization
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import optuna
    from optuna.trial import Trial
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("Optuna not installed. Please run: uv sync --all-extras")
    print("Or: uv add optuna optuna-dashboard")
    sys.exit(1)

from otcfm.config import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig
from otcfm.datasets import (
    load_caltech101, load_scene15, load_noisy_mnist,
    load_bdgp, load_cub, load_reuters, load_synthetic,
    load_handwritten, load_coil20, load_nus_wide,
    create_dataloader, MultiViewDataset
)
from otcfm.ot_cfm import OTCFM
from otcfm.trainer import Trainer
from otcfm.metrics import evaluate_clustering


# Dataset loader mapping
DATASET_LOADERS = {
    'caltech101': load_caltech101,
    'scene15': load_scene15,
    'noisy_mnist': load_noisy_mnist,
    'bdgp': load_bdgp,
    'cub': load_cub,
    'reuters': load_reuters,
    'synthetic': load_synthetic,
    'handwritten': load_handwritten,
    'coil20': load_coil20,
    'nus-wide': load_nus_wide,
}


class OptunaHyperparameterTuner:
    """Optuna-based hyperparameter tuning for OT-CFM"""
    
    def __init__(
        self,
        dataset_name: str,
        data_root: str = "./data",
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        device: str = "cpu",
        seed: int = 42,
        tuning_epochs: int = 50,
        save_dir: str = "config",
    ):
        """
        Initialize the tuner
        
        Args:
            dataset_name: Name of the dataset
            data_root: Root directory for data
            n_trials: Number of Optuna trials
            timeout: Maximum time for tuning in seconds
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            study_name: Name for the Optuna study
            storage: Optuna storage URL (e.g., sqlite:///optuna.db)
            device: PyTorch device
            seed: Random seed
            tuning_epochs: Number of epochs for each trial (less than full training)
            save_dir: Directory to save tuned parameters
        """
        self.dataset_name = dataset_name.lower()
        self.data_root = data_root
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study_name = study_name or f"otcfm_{self.dataset_name}"
        self.storage = storage
        self.device = device
        self.seed = seed
        self.tuning_epochs = tuning_epochs
        self.save_dir = Path(save_dir)
        
        # Load dataset once
        self._load_dataset()
        
    def _load_dataset(self):
        """Load the dataset"""
        if self.dataset_name not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        loader = DATASET_LOADERS[self.dataset_name]
        
        if self.dataset_name == 'synthetic':
            data = loader(n_samples=1000, n_clusters=10)
            self.views = data['views']
            self.labels = data['labels']
        else:
            result = loader(self.data_root)
            if isinstance(result, tuple):
                self.views, self.labels = result
            else:
                self.views = result['views']
                self.labels = result['labels']
        
        self.view_dims = [v.shape[1] for v in self.views]
        self.n_clusters = len(np.unique(self.labels))
        self.n_samples = len(self.labels)
        
        print(f"Loaded {self.dataset_name}:")
        print(f"  Samples: {self.n_samples}")
        print(f"  Views: {len(self.views)}")
        print(f"  View dims: {self.view_dims}")
        print(f"  Clusters: {self.n_clusters}")
    
    def _suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial"""
        params = {
            # Model architecture
            "latent_dim": trial.suggest_categorical("latent_dim", [64, 128, 256, 512]),
            "hidden_dims": trial.suggest_categorical("hidden_dims", [
                "[256, 128]",
                "[512, 256]",
                "[512, 256, 128]",
                "[1024, 512, 256]"
            ]),
            "flow_hidden_dim": trial.suggest_categorical("flow_hidden_dim", [128, 256, 512]),
            "flow_num_layers": trial.suggest_int("flow_num_layers", 2, 6),
            "time_dim": trial.suggest_categorical("time_dim", [32, 64, 128]),
            "ode_steps": trial.suggest_int("ode_steps", 5, 20),
            
            # Loss weights
            "lambda_gw": trial.suggest_float("lambda_gw", 0.01, 1.0, log=True),
            "lambda_cluster": trial.suggest_float("lambda_cluster", 0.1, 2.0, log=True),
            "lambda_recon": trial.suggest_float("lambda_recon", 0.5, 2.0),
            "lambda_contrastive": trial.suggest_float("lambda_contrastive", 0.01, 0.5, log=True),
            
            # Training
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            
            # Kernel
            "kernel_type": trial.suggest_categorical("kernel_type", ["rbf", "cosine", "linear"]),
            "kernel_gamma": trial.suggest_float("kernel_gamma", 0.1, 10.0, log=True),
        }
        
        return params
    
    def _create_model_and_train(self, params: Dict[str, Any]) -> float:
        """Create model with given params and train, return best ACC"""
        # Parse hidden_dims string
        hidden_dims = eval(params["hidden_dims"])
        
        # Create dataset
        dataset = MultiViewDataset(views=self.views, labels=self.labels)
        train_loader = create_dataloader(dataset, params["batch_size"], shuffle=True)
        
        # Create model
        model = OTCFM(
            view_dims=self.view_dims,
            latent_dim=params["latent_dim"],
            hidden_dims=hidden_dims,
            num_clusters=self.n_clusters,
            flow_hidden_dim=params["flow_hidden_dim"],
            flow_num_layers=params["flow_num_layers"],
            time_dim=params["time_dim"],
            ode_steps=params["ode_steps"],
            sigma_min=1e-4,
            kernel_type=params["kernel_type"],
            kernel_gamma=params["kernel_gamma"],
            lambda_gw=params["lambda_gw"],
            lambda_cluster=params["lambda_cluster"],
            lambda_recon=params["lambda_recon"],
            lambda_contrastive=params["lambda_contrastive"],
            dropout=params["dropout"]
        )
        
        # Create training config
        training_config = TrainingConfig(
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            epochs=self.tuning_epochs,
            device=self.device,
        )
        
        # Create trainer (use temp directory)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                config=training_config,
                experiment_dir=tmpdir,
                device=self.device,
                verbose=False  # Quiet mode for tuning
            )
            
            # Train
            results = trainer.train(train_loader, self.labels)
        
        return results['best']['acc']
    
    def objective(self, trial: Trial) -> float:
        """Optuna objective function"""
        # Set seed for reproducibility
        torch.manual_seed(self.seed + trial.number)
        np.random.seed(self.seed + trial.number)
        
        try:
            params = self._suggest_hyperparameters(trial)
            acc = self._create_model_and_train(params)
            return acc
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return worst score on failure
    
    def run(self) -> Dict[str, Any]:
        """Run the hyperparameter tuning"""
        print(f"\n{'='*60}")
        print(f"Starting Optuna Hyperparameter Tuning")
        print(f"Dataset: {self.dataset_name}")
        print(f"Trials: {self.n_trials}")
        print(f"Epochs per trial: {self.tuning_epochs}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Create sampler and pruner
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        # Create or load study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\n{'='*60}")
        print(f"Tuning Complete!")
        print(f"Best ACC: {best_value:.4f}")
        print(f"{'='*60}")
        print("\nBest Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Save results
        self._save_results(study, best_params, best_value)
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
        }
    
    def _save_results(self, study, best_params: Dict, best_value: float):
        """Save tuning results"""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        tuned_params_path = self.save_dir / "tuned_params.json"
        
        # Load existing params or create new
        if tuned_params_path.exists():
            with open(tuned_params_path, 'r') as f:
                all_params = json.load(f)
        else:
            all_params = {}
        
        # Update with this dataset's params
        all_params[self.dataset_name] = {
            "params": best_params,
            "best_acc": best_value,
            "tuned_at": datetime.now().isoformat(),
            "n_trials": len(study.trials),
        }
        
        with open(tuned_params_path, 'w') as f:
            json.dump(all_params, f, indent=2)
        
        print(f"\nTuned parameters saved to: {tuned_params_path}")
        
        # Save full study results
        study_path = self.save_dir / f"{self.dataset_name}_study.json"
        study_results = {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ]
        }
        
        with open(study_path, 'w') as f:
            json.dump(study_results, f, indent=2)
        
        print(f"Full study saved to: {study_path}")


def load_tuned_params(dataset_name: str, config_dir: str = "config") -> Optional[Dict[str, Any]]:
    """
    Load tuned parameters for a dataset
    
    Args:
        dataset_name: Name of the dataset
        config_dir: Directory containing tuned_params.json
    
    Returns:
        Dictionary of tuned parameters or None if not found
    """
    tuned_params_path = Path(config_dir) / "tuned_params.json"
    
    if not tuned_params_path.exists():
        print(f"No tuned parameters found at {tuned_params_path}")
        return None
    
    with open(tuned_params_path, 'r') as f:
        all_params = json.load(f)
    
    dataset_key = dataset_name.lower()
    if dataset_key not in all_params:
        print(f"No tuned parameters for dataset: {dataset_name}")
        print(f"Available datasets: {list(all_params.keys())}")
        return None
    
    return all_params[dataset_key]["params"]


def apply_tuned_params(config: ExperimentConfig, tuned_params: Dict[str, Any]) -> ExperimentConfig:
    """
    Apply tuned parameters to an experiment config
    
    Args:
        config: Base experiment config
        tuned_params: Dictionary of tuned parameters
    
    Returns:
        Updated experiment config
    """
    # Model params
    if "latent_dim" in tuned_params:
        config.model.latent_dim = tuned_params["latent_dim"]
    if "hidden_dims" in tuned_params:
        config.model.hidden_dims = eval(tuned_params["hidden_dims"])
    if "flow_hidden_dim" in tuned_params:
        config.model.flow_hidden_dim = tuned_params["flow_hidden_dim"]
    if "flow_num_layers" in tuned_params:
        config.model.flow_num_layers = tuned_params["flow_num_layers"]
    if "time_dim" in tuned_params:
        config.model.time_dim = tuned_params["time_dim"]
    if "ode_steps" in tuned_params:
        config.model.ode_steps = tuned_params["ode_steps"]
    
    # Loss weights
    if "lambda_gw" in tuned_params:
        config.model.lambda_gw = tuned_params["lambda_gw"]
    if "lambda_cluster" in tuned_params:
        config.model.lambda_cluster = tuned_params["lambda_cluster"]
    if "lambda_recon" in tuned_params:
        config.model.lambda_recon = tuned_params["lambda_recon"]
    if "lambda_contrastive" in tuned_params:
        config.model.lambda_contrastive = tuned_params["lambda_contrastive"]
    
    # Training params
    if "learning_rate" in tuned_params:
        config.training.learning_rate = tuned_params["learning_rate"]
    if "weight_decay" in tuned_params:
        config.training.weight_decay = tuned_params["weight_decay"]
    if "batch_size" in tuned_params:
        config.training.batch_size = tuned_params["batch_size"]
    
    # Model params continued
    if "dropout" in tuned_params:
        config.model.dropout = tuned_params["dropout"]
    if "kernel_type" in tuned_params:
        config.model.kernel_type = tuned_params["kernel_type"]
    if "kernel_gamma" in tuned_params:
        config.model.kernel_gamma = tuned_params["kernel_gamma"]
    
    return config


def main():
    parser = argparse.ArgumentParser(description='OT-CFM Optuna Hyperparameter Tuning')
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., Handwritten, Scene15, Coil20)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Maximum tuning time in seconds')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs')
    parser.add_argument('--tuning_epochs', type=int, default=50,
                        help='Number of epochs per trial')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu, auto-detect if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='config',
                        help='Directory to save tuned parameters')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
        print(f"Auto-detected device: {args.device}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run tuning
    tuner = OptunaHyperparameterTuner(
        dataset_name=args.dataset,
        data_root=args.data_root,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        device=args.device,
        seed=args.seed,
        tuning_epochs=args.tuning_epochs,
        save_dir=args.save_dir,
        storage=args.storage,
    )
    
    results = tuner.run()
    
    print(f"\n{'='*60}")
    print("Tuning Summary")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Best ACC: {results['best_value']:.4f}")
    print(f"Total trials: {results['n_trials']}")
    print(f"\nTo use tuned parameters:")
    print(f"  uv run python scripts/run_experiment.py --dataset {args.dataset} --use_tuned")


if __name__ == "__main__":
    main()
