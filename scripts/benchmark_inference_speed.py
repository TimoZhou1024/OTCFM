"""
Benchmark script: Inference speed comparison between OPTION and DCG
Measures wall-clock time for missing view imputation across different settings
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from otcfm.config import ModelConfig, TrainingConfig, get_device
from otcfm.datasets import (
    load_scene15, load_handwritten, load_coil20, load_bdgp, load_cub, load_nus_wide,
    create_dataloader, MultiViewDataset
)
from otcfm.ot_cfm import OTCFM
from otcfm.models import ODESolver


class DCGImputation:
    """Simplified DCG-like diffusion baseline for speed comparison"""
    def __init__(self, latent_dim=128, num_diffusion_steps=100, device='cuda'):
        self.latent_dim = latent_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
        
        # Simple denoising network
        self.denoiser = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),  # +1 for time
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        ).to(device)
    
    def impute_missing_views(self, z_available, num_samples=256):
        """
        Simulate DCG diffusion-based imputation
        num_diffusion_steps iterations (default 100)
        """
        batch_size = z_available.shape[0]
        
        # Start from Gaussian noise
        z_t = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        # Iterative denoising
        for step in range(self.num_diffusion_steps):
            # Create time embedding: shape [batch_size, 1]
            t = torch.full((batch_size, 1), step / self.num_diffusion_steps, device=self.device)
            
            # Concatenate with time embedding: [batch_size, latent_dim + 1]
            z_with_t = torch.cat([z_t, t], dim=-1)
            
            # Denoising step
            noise = self.denoiser(z_with_t)
            alpha = 1.0 - (step + 1) / self.num_diffusion_steps
            z_t = alpha * z_t + (1 - alpha) * noise
        
        return z_t


class OptimizedODESolver(ODESolver):
    """ODE solver for OPTION (with 10 steps)"""
    def __init__(self, num_steps=10, **kwargs):
        super().__init__(**kwargs)
        self.num_steps = num_steps


def benchmark_imputation(
    model: OTCFM,
    data_loader,
    num_iterations: int = 10,
    ode_steps: int = 10,
    device: str = 'cuda',
    dataset_name: str = 'Scene15'
) -> Dict:
    """Benchmark OPTION inference"""
    
    model.eval()
    times = []
    
    with torch.no_grad():
        for iteration in range(num_iterations):
            batch_idx = 0
            for batch_data in data_loader:
                if batch_idx >= 1:  # Use first batch only
                    break
                
                # Prepare batch (handle MultiViewDataset format)
                if isinstance(batch_data, dict):
                    if 'views' in batch_data:
                        # MultiViewDataset format: {'views': [tensor, tensor, ...], 'mask': ..., 'label': ...}
                        views = [v.float().to(device) for v in batch_data['views']]
                        mask = batch_data.get('mask', None)
                        if mask is not None:
                            mask = mask.to(device)
                    else:
                        # Legacy format: {'view_0': ..., 'view_1': ...}
                        num_views = sum(1 for k in batch_data.keys() if k.startswith('view_'))
                        views = [batch_data[f'view_{v}'].to(device) for v in range(num_views)]
                        mask = batch_data.get('mask', None)
                        if mask is not None:
                            mask = mask.to(device)
                else:
                    views = [v.to(device) for v in batch_data[:-1]]
                    mask = batch_data[-1].to(device) if len(batch_data) > len(views) else None
                
                # Warm up first iteration
                if iteration == 0 and batch_idx == 0:
                    _ = model(views, mask)
                    continue
                
                batch_size = views[0].shape[0]
                
                # Time the inference
                torch.cuda.synchronize() if device == 'cuda' else None
                start_time = time.time()
                
                with torch.no_grad():
                    _ = model(views, mask)
                
                torch.cuda.synchronize() if device == 'cuda' else None
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000  # Convert to ms
                times.append(elapsed)
                batch_idx += 1
    
    return {
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'times': times
    }


def benchmark_dcg_baseline(
    latent_dim: int,
    batch_size: int = 256,
    num_iterations: int = 10,
    diffusion_steps: int = 100,
    device: str = 'cuda'
) -> Dict:
    """Benchmark DCG-like diffusion baseline"""
    
    dcg = DCGImputation(latent_dim=latent_dim, num_diffusion_steps=diffusion_steps, device=device)
    dcg.denoiser.eval()
    
    times = []
    
    with torch.no_grad():
        for iteration in range(num_iterations):
            # Warm up
            if iteration == 0:
                z_dummy = torch.randn(batch_size, latent_dim, device=device)
                _ = dcg.impute_missing_views(z_dummy, num_samples=batch_size)
                continue
            
            z_available = torch.randn(batch_size, latent_dim, device=device)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                _ = dcg.impute_missing_views(z_available, num_samples=batch_size)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.time()
            
            elapsed = (end_time - start_time) * 1000  # Convert to ms
            times.append(elapsed)
    
    return {
        'mean_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'times': times,
        'speedup_vs_dcg': 1.0  # Baseline
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference speed')
    parser.add_argument('--dataset', type=str, default='scene15', 
                        choices=['scene15', 'handwritten', 'coil20', 'bdgp', 'cub', 'nus-wide'])
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing .mat files')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_iterations', type=int, default=10, help='Number of warmup + inference iterations')
    parser.add_argument('--ode_steps', type=int, default=10, help='ODE steps for OPTION')
    parser.add_argument('--diffusion_steps', type=int, default=100, help='Diffusion steps for DCG')
    parser.add_argument('--device', type=str, default=get_device())
    parser.add_argument('--output_dir', type=str, default='benchmark_results')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained OPTION model')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 80)
    print(f"Inference Speed Benchmark: OPTION vs DCG")
    print(f"=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"OPTION ODE Steps: {args.ode_steps}")
    print(f"DCG Diffusion Steps: {args.diffusion_steps}")
    print()
    
    # Load dataset
    print(f"Loading {args.dataset}...")
    dataset_loaders = {
        'scene15': lambda: load_scene15(args.data_dir),
        'handwritten': lambda: load_handwritten(args.data_dir),
        'coil20': lambda: load_coil20(args.data_dir),
        'bdgp': lambda: load_bdgp(args.data_dir),
        'cub': lambda: load_cub(args.data_dir),
        'nus-wide': lambda: load_nus_wide(args.data_dir),
    }
    
    try:
        X, labels = dataset_loaders[args.dataset]()
    except FileNotFoundError as e:
        print(f"ERROR: Dataset file not found. {e}")
        print(f"Make sure .mat files are in {args.data_dir}/")
        return
    view_dims = [v.shape[1] for v in X]
    num_clusters = len(np.unique(labels))
    
    print(f"  Views: {len(X)}, Dimensions: {view_dims}")
    print(f"  Num clusters: {num_clusters}")
    print()
    
    # Create data loader
    dataset = MultiViewDataset(X, labels)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize OPTION model
    print("Initializing OPTION model...")
    model_config = ModelConfig(
        latent_dim=128,
        num_clusters=num_clusters,
        ode_steps=args.ode_steps
    )
    
    option_model = OTCFM(
        view_dims=view_dims,
        latent_dim=model_config.latent_dim,
        num_clusters=model_config.num_clusters,
        ode_steps=args.ode_steps,
        is_aligned=True
    )
    
    # Load pretrained model if available
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pretrained model from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            option_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            option_model.load_state_dict(checkpoint)
    
    option_model = option_model.to(args.device)
    print()
    
    # Benchmark OPTION
    print(f"Benchmarking OPTION (ODE steps={args.ode_steps})...")
    option_results = benchmark_imputation(
        option_model,
        data_loader,
        num_iterations=args.num_iterations,
        ode_steps=args.ode_steps,
        device=args.device,
        dataset_name=args.dataset
    )
    print(f"  Mean time: {option_results['mean_time_ms']:.2f} ± {option_results['std_time_ms']:.2f} ms")
    print()
    
    # Benchmark DCG baseline
    print(f"Benchmarking DCG baseline (diffusion steps={args.diffusion_steps})...")
    dcg_results = benchmark_dcg_baseline(
        latent_dim=model_config.latent_dim,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        diffusion_steps=args.diffusion_steps,
        device=args.device
    )
    print(f"  Mean time: {dcg_results['mean_time_ms']:.2f} ± {dcg_results['std_time_ms']:.2f} ms")
    print()
    
    # Compute speedup
    speedup = dcg_results['mean_time_ms'] / option_results['mean_time_ms']
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"OPTION (10 ODE steps):     {option_results['mean_time_ms']:7.2f} ms")
    print(f"DCG ({args.diffusion_steps} diffusion steps): {dcg_results['mean_time_ms']:7.2f} ms")
    print(f"Speedup: {speedup:.1f}×")
    print("=" * 80)
    print()
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'view_dims': view_dims,
        'num_clusters': num_clusters,
        'ode_steps': args.ode_steps,
        'diffusion_steps': args.diffusion_steps,
        'device': args.device,
        'option': {
            'mean_time_ms': float(option_results['mean_time_ms']),
            'std_time_ms': float(option_results['std_time_ms']),
            'min_time_ms': float(option_results['min_time_ms']),
            'max_time_ms': float(option_results['max_time_ms']),
        },
        'dcg': {
            'mean_time_ms': float(dcg_results['mean_time_ms']),
            'std_time_ms': float(dcg_results['std_time_ms']),
            'min_time_ms': float(dcg_results['min_time_ms']),
            'max_time_ms': float(dcg_results['max_time_ms']),
        },
        'speedup': float(speedup)
    }
    
    results_file = output_dir / f"benchmark_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    main()
