"""
t-SNE Visualization for OT-CFM Training Process
Generates publication-quality visualizations showing latent space evolution during training.

Usage:
    uv run python scripts/run_tsne_visualization.py --dataset Handwritten --epochs 100
    uv run python scripts/run_tsne_visualization.py --dataset Scene15 --epochs 150 --checkpoints 0,25,50,75,100,150
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.manifold import TSNE

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from otcfm.config import (
    ExperimentConfig, ModelConfig, TrainingConfig, DataConfig,
    get_default_config
)
from otcfm.datasets import (
    load_scene15, load_handwritten, load_coil20, load_noisy_mnist,
    load_caltech101, load_bdgp, MultiViewDataset, create_dataloader
)
from otcfm.ot_cfm import OTCFM
from otcfm.trainer import Trainer
from otcfm.metrics import evaluate_clustering
from otcfm.utils import set_seed

DATASET_LOADERS = {
    'scene15': load_scene15,
    'handwritten': load_handwritten,
    'coil20': load_coil20,
    'noisymnist': load_noisy_mnist,
    'noisy_mnist': load_noisy_mnist,
    'caltech101': load_caltech101,
    'bdgp': load_bdgp,
}

# Publication-quality color palette
CLUSTER_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
    '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'
]


def get_embeddings_and_predictions(model, dataloader, device):
    """Extract consensus embeddings and predictions from the model"""
    model.eval()
    all_embeddings = []
    all_predictions = []
    all_indices = []
    
    with torch.no_grad():
        for batch in dataloader:
            views = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            indices = batch['index']
            
            outputs = model(views, mask, return_all=True)
            
            # Use consensus embedding
            if outputs['consensus'] is not None:
                embeddings = outputs['consensus'].cpu().numpy()
            else:
                # Fallback: average of latents
                latents = [z.cpu().numpy() for z in outputs['latents']]
                embeddings = np.mean(latents, axis=0)
            
            # Get predictions
            predictions = outputs['assignments'].cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_predictions.append(predictions)
            if isinstance(indices, torch.Tensor):
                all_indices.append(indices.numpy())
            else:
                all_indices.append(np.array(indices))
    
    # Sort by original indices
    all_embeddings = np.vstack(all_embeddings)
    all_predictions = np.concatenate(all_predictions)
    all_indices = np.concatenate(all_indices)
    
    order = np.argsort(all_indices)
    return all_embeddings[order], all_predictions[order]


def train_with_checkpoints(
    model, train_loader, labels, device, config, 
    checkpoint_epochs, output_dir, verbose=True
):
    """Train model and save embeddings at specified checkpoints"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Store embeddings at checkpoints
    embeddings_dict = {}
    metrics_dict = {}
    
    total_epochs = config.training.epochs
    pretrain_epochs = min(20, total_epochs // 5)  # 20% for pretraining
    
    # Create full dataloader for embedding extraction
    full_dataset = train_loader.dataset
    full_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=256, shuffle=False
    )
    
    # Epoch 0 - initial random embeddings
    if 0 in checkpoint_epochs:
        emb, preds = get_embeddings_and_predictions(model, full_loader, device)
        embeddings_dict[0] = emb
        
        # Get metrics
        metrics = evaluate_clustering(labels, preds)
        metrics_dict[0] = metrics
        if verbose:
            print(f"Epoch 0: ACC={metrics['acc']:.4f}, NMI={metrics['nmi']:.4f}")
    
    # Phase 1: Pretraining (reconstruction only)
    if verbose:
        print(f"\nPhase 1: Reconstruction pretraining ({pretrain_epochs} epochs)...")
    
    for epoch in range(pretrain_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            views = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # Reconstruction only
            outputs = model(views, mask, return_all=True)
            recon_loss = 0
            for v_idx in range(len(views)):
                recon_loss += torch.nn.functional.mse_loss(
                    outputs['reconstructions'][v_idx], views[v_idx]
                )
            recon_loss = recon_loss / len(views)
            
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += recon_loss.item()
            num_batches += 1
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Pretrain epoch {epoch+1}/{pretrain_epochs}, Loss: {total_loss/num_batches:.4f}")
    
    # Initialize clustering centroids
    if verbose:
        print("\nInitializing clustering centroids...")
    model.init_clustering(full_loader, device)
    
    # Phase 2: Full training
    main_epochs = total_epochs - pretrain_epochs
    if verbose:
        print(f"\nPhase 2: Full training ({main_epochs} epochs)...")
    
    pbar = tqdm(range(main_epochs), desc="Training")
    for epoch in pbar:
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            views = [v.to(device) for v in batch['views']]
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # Use compute_loss for full training
            loss, loss_dict = model.compute_loss(views, mask)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        current_epoch = pretrain_epochs + epoch + 1
        
        # Save checkpoint if needed
        if current_epoch in checkpoint_epochs:
            emb, preds = get_embeddings_and_predictions(model, full_loader, device)
            embeddings_dict[current_epoch] = emb
            
            # Get metrics
            metrics = evaluate_clustering(labels, preds)
            metrics_dict[current_epoch] = metrics
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ACC': f'{metrics["acc"]:.4f}',
                'NMI': f'{metrics["nmi"]:.4f}'
            })
            
            if verbose:
                print(f"\nEpoch {current_epoch}: ACC={metrics['acc']:.4f}, NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}")
    
    return embeddings_dict, metrics_dict


def compute_tsne(embeddings_dict, perplexity=30, random_state=42, verbose=True):
    """Compute t-SNE for all checkpoints using consistent random state"""
    tsne_dict = {}
    
    for epoch, emb in embeddings_dict.items():
        if verbose:
            print(f"Computing t-SNE for epoch {epoch}...")
        
        # Use same t-SNE parameters for consistency
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(emb) // 4),
            random_state=random_state,
            max_iter=1000,
            init='pca'
        )
        tsne_emb = tsne.fit_transform(emb)
        tsne_dict[epoch] = tsne_emb
    
    return tsne_dict


def plot_tsne_evolution(
    tsne_dict, labels, metrics_dict, 
    dataset_name, save_path, 
    figsize=(15, 4), dpi=300
):
    """Generate publication-quality t-SNE evolution plot"""
    
    epochs = sorted(tsne_dict.keys())
    n_plots = len(epochs)
    n_clusters = len(np.unique(labels))
    
    # Use colorblind-friendly palette
    if n_clusters <= len(CLUSTER_COLORS):
        colors = CLUSTER_COLORS[:n_clusters]
    else:
        cmap = plt.cm.get_cmap('tab20')
        colors = [cmap(i / n_clusters) for i in range(n_clusters)]
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    for idx, (epoch, ax) in enumerate(zip(epochs, axes)):
        tsne_emb = tsne_dict[epoch]
        metrics = metrics_dict.get(epoch, {})
        
        # Plot each cluster
        for c in range(n_clusters):
            mask = labels == c
            ax.scatter(
                tsne_emb[mask, 0],
                tsne_emb[mask, 1],
                c=[colors[c]],
                s=8,
                alpha=0.7,
                edgecolors='none'
            )
        
        # Title with metrics
        acc = metrics.get('acc', 0) * 100
        nmi = metrics.get('nmi', 0) * 100
        
        if epoch == 0:
            ax.set_title(f'Initial\n(ACC: {acc:.1f}%)', fontsize=11)
        else:
            ax.set_title(f'Epoch {epoch}\n(ACC: {acc:.1f}%)', fontsize=11)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    
    plt.suptitle(f't-SNE Visualization of Latent Space Evolution ({dataset_name})', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    
    # Also save as PDF for LaTeX
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()


def plot_tsne_single(
    tsne_emb, labels, epoch, metrics, 
    dataset_name, save_path, 
    figsize=(5, 5), dpi=300
):
    """Generate single t-SNE plot for a specific epoch"""
    
    n_clusters = len(np.unique(labels))
    
    if n_clusters <= len(CLUSTER_COLORS):
        colors = CLUSTER_COLORS[:n_clusters]
    else:
        cmap = plt.cm.get_cmap('tab20')
        colors = [cmap(i / n_clusters) for i in range(n_clusters)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each cluster
    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(
            tsne_emb[mask, 0],
            tsne_emb[mask, 1],
            c=[colors[c]],
            s=15,
            alpha=0.7,
            edgecolors='none',
            label=f'Class {c}'
        )
    
    acc = metrics.get('acc', 0) * 100
    nmi = metrics.get('nmi', 0) * 100
    
    if epoch == 0:
        ax.set_title(f'{dataset_name} - Initial State\nACC: {acc:.1f}%, NMI: {nmi:.1f}%', fontsize=12)
    else:
        ax.set_title(f'{dataset_name} - Epoch {epoch}\nACC: {acc:.1f}%, NMI: {nmi:.1f}%', fontsize=12)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='t-SNE Visualization for OT-CFM')
    parser.add_argument('--dataset', type=str, default='Handwritten',
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total training epochs')
    parser.add_argument('--checkpoints', type=str, default=None,
                        help='Comma-separated checkpoint epochs (e.g., "0,25,50,75,100")')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='figures',
                        help='Output directory')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Parse checkpoints
    if args.checkpoints:
        checkpoint_epochs = [int(x) for x in args.checkpoints.split(',')]
    else:
        # Default: 4-6 checkpoints evenly distributed
        n_checkpoints = 5
        checkpoint_epochs = [0] + [
            int(args.epochs * i / (n_checkpoints - 1)) 
            for i in range(1, n_checkpoints)
        ]
    
    print(f"{'='*70}")
    print(f"t-SNE Visualization for OT-CFM")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Checkpoints: {checkpoint_epochs}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}")
    
    # Load dataset
    dataset_name = args.dataset.lower()
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    loader = DATASET_LOADERS[dataset_name]
    views, labels = loader(data_dir='./data')
    
    num_clusters = len(np.unique(labels))
    view_dims = [v.shape[1] for v in views]
    n_samples = len(labels)
    
    print(f"Loaded {n_samples} samples, {len(views)} views, {num_clusters} clusters")
    print(f"View dimensions: {view_dims}")
    
    # Create dataloader
    dataset = MultiViewDataset(views, labels)
    train_loader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create config
    config = get_default_config()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.model.num_clusters = num_clusters
    
    # Create model
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=config.model.latent_dim,
        hidden_dims=config.model.hidden_dims,
        num_clusters=num_clusters,
        flow_hidden_dim=config.model.flow_hidden_dim,
        lambda_gw=config.model.lambda_gw,
        lambda_cluster=config.model.lambda_cluster,
        lambda_recon=config.model.lambda_recon,
        lambda_contrastive=config.model.lambda_contrastive,
        is_aligned=True
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train and collect embeddings
    print("\nTraining and collecting embeddings...")
    embeddings_dict, metrics_dict = train_with_checkpoints(
        model, train_loader, labels, device, config,
        checkpoint_epochs, output_dir, verbose=True
    )
    
    # Compute t-SNE
    print("\nComputing t-SNE projections...")
    tsne_dict = compute_tsne(embeddings_dict, perplexity=args.perplexity, verbose=True)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Main evolution plot
    main_plot_path = output_dir / f'{args.dataset.lower()}_tsne_evolution.png'
    plot_tsne_evolution(
        tsne_dict, labels, metrics_dict,
        args.dataset, str(main_plot_path),
        figsize=(3 * len(checkpoint_epochs), 3.5)
    )
    
    # Individual plots for each checkpoint
    for epoch in checkpoint_epochs:
        single_plot_path = output_dir / f'{args.dataset.lower()}_tsne_epoch{epoch}.png'
        plot_tsne_single(
            tsne_dict[epoch], labels, epoch, metrics_dict[epoch],
            args.dataset, str(single_plot_path)
        )
    
    # Save metrics summary
    summary = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'checkpoints': checkpoint_epochs,
        'seed': args.seed,
        'metrics': {str(k): v for k, v in metrics_dict.items()}
    }
    
    summary_path = output_dir / f'{args.dataset.lower()}_tsne_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}")
    for epoch in checkpoint_epochs:
        m = metrics_dict[epoch]
        print(f"Epoch {epoch:3d}: ACC={m['acc']*100:.1f}%, NMI={m['nmi']*100:.1f}%, ARI={m['ari']*100:.1f}%")
    
    print(f"\n✅ Visualizations saved to: {output_dir}")
    print(f"   - Main plot: {main_plot_path.name}")
    print(f"   - Individual plots: {args.dataset.lower()}_tsne_epoch*.png")


if __name__ == '__main__':
    main()
