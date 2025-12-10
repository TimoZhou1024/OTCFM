"""
Training pipeline for OT-CFM
Implements alternating optimization as described in Algorithm 1
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm

from .config import TrainingConfig, ModelConfig, ExperimentConfig
from .datasets import MultiViewDataset
from .ot_cfm import OTCFM
from .metrics import evaluate_clustering, MetricTracker


class Trainer:
    """
    Full training pipeline for OT-CFM with alternating optimization
    """
    
    def __init__(
        self,
        model: OTCFM,
        config: TrainingConfig,
        experiment_dir: str,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Metric tracker
        self.tracker = MetricTracker()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config"""
        if self.config.optimizer == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if self.config.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 4,
                gamma=0.5
            )
        elif self.config.scheduler == 'none':
            return None
        else:
            return None
    
    def train(
        self,
        train_loader: DataLoader,
        labels: np.ndarray,
        val_loader: Optional[DataLoader] = None,
        val_labels: Optional[np.ndarray] = None,
        ablation_mode: str = "full",
        pretrain_epochs: int = 20  # 预训练轮数
    ) -> Dict:
        """
        Full training loop with alternating optimization
        
        Args:
            train_loader: Training data loader
            labels: Ground truth labels for evaluation
            val_loader: Optional validation loader
            val_labels: Optional validation labels
            ablation_mode: Ablation setting
            pretrain_epochs: Number of pretraining epochs for encoder-decoder
        
        Returns:
            Final metrics dictionary
        """
        # Phase 1: Pretrain encoder-decoder (重建任务)
        recon_epochs = pretrain_epochs // 2 if pretrain_epochs > 10 else pretrain_epochs
        dec_epochs = pretrain_epochs - recon_epochs
        
        if recon_epochs > 0:
            print(f"Phase 1: Reconstruction pretraining for {recon_epochs} epochs...")
            for epoch in range(recon_epochs):
                self.model.train()
                total_recon_loss = 0
                num_batches = 0
                
                for batch in train_loader:
                    views = [v.to(self.device) for v in batch['views']]
                    mask = batch['mask'].to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    # Only compute reconstruction loss
                    outputs = self.model(views, mask, return_all=True)
                    
                    recon_loss = 0
                    for v_idx in range(len(views)):
                        recon_loss += F.mse_loss(
                            outputs['reconstructions'][v_idx], 
                            views[v_idx]
                        )
                    recon_loss = recon_loss / len(views)
                    
                    loss = recon_loss
                    loss.backward()
                    
                    if self.config.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.clip_grad_norm
                        )
                    
                    self.optimizer.step()
                    total_recon_loss += recon_loss.item()
                    num_batches += 1
                
                if (epoch + 1) % 5 == 0:
                    avg_loss = total_recon_loss / num_batches
                    print(f"  Recon epoch {epoch+1}/{recon_epochs}, Loss: {avg_loss:.4f}")
        
        # Initialize clustering centroids after reconstruction pretraining
        print("Initializing clustering centroids...")
        self.model.init_clustering(train_loader, self.device)
        
        # Phase 2: Single-View DEC pretraining (单视图聚类)
        if dec_epochs > 0:
            print(f"Phase 2: Single-View DEC pretraining for {dec_epochs} epochs...")
            for epoch in range(dec_epochs):
                self.model.train()
                total_loss = 0
                total_dec_loss = 0
                total_recon_loss = 0
                num_batches = 0
                
                for batch in train_loader:
                    views = [v.to(self.device) for v in batch['views']]
                    mask = batch['mask'].to(self.device)
                    
                    self.optimizer.zero_grad()
                    
                    outputs = self.model(views, mask, return_all=True)
                    latents = outputs['latents']
                    
                    # Reconstruction loss
                    recon_loss = 0
                    for v_idx in range(len(views)):
                        recon_loss += F.mse_loss(
                            outputs['reconstructions'][v_idx], 
                            views[v_idx]
                        )
                    recon_loss = recon_loss / len(views)
                    
                    # Single-View DEC loss: cluster each view independently
                    # This builds cluster-separable latent space BEFORE cross-view alignment
                    sv_dec_loss = 0
                    for z_v in latents:
                        # Compute soft assignments for this view
                        q_v, p_v = self.model.clustering(z_v)
                        # KL divergence loss (DEC objective)
                        kl_loss = (p_v * torch.log((p_v + 1e-8) / (q_v + 1e-8))).sum(dim=1).mean()
                        sv_dec_loss += kl_loss
                    sv_dec_loss = sv_dec_loss / len(latents)
                    
                    # Combined loss: reconstruction + single-view DEC
                    loss = recon_loss + 1.0 * sv_dec_loss
                    loss.backward()
                    
                    if self.config.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.clip_grad_norm
                        )
                    
                    self.optimizer.step()
                    total_loss += loss.item()
                    total_dec_loss += sv_dec_loss.item()
                    total_recon_loss += recon_loss.item()
                    num_batches += 1
                
                if (epoch + 1) % 5 == 0:
                    avg_loss = total_loss / num_batches
                    avg_dec = total_dec_loss / num_batches
                    avg_recon = total_recon_loss / num_batches
                    print(f"  SV-DEC epoch {epoch+1}/{dec_epochs}, Loss: {avg_loss:.4f} (DEC: {avg_dec:.4f}, Recon: {avg_recon:.4f})")
            
            # Re-initialize centroids after SV-DEC training for better starting point
            print("Re-initializing centroids after SV-DEC pretraining...")
            self.model.init_clustering(train_loader, self.device)
        
        # Training loop
        best_metrics = {}
        training_history = []
        
        pbar = tqdm(range(self.config.epochs), desc="Training")
        for epoch in pbar:
            self.epoch = epoch
            
            # E-step: Update clustering assignments (at specified frequency)
            if epoch % self.config.cluster_update_freq == 0 and epoch > 0:
                self._update_clustering(train_loader)
            
            # M-step: Update network parameters
            train_metrics = self._train_epoch(train_loader, ablation_mode)
            
            # Evaluate
            eval_metrics = self._evaluate(train_loader, labels)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track metrics
            all_metrics = {**train_metrics, **eval_metrics}
            self.tracker.update(all_metrics)
            training_history.append(all_metrics)
            
            # Update best model
            if eval_metrics['acc'] > self.best_metric:
                self.best_metric = eval_metrics['acc']
                best_metrics = eval_metrics.copy()
                self._save_checkpoint('best.pth')
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{train_metrics['loss']:.4f}",
                'acc': f"{eval_metrics['acc']:.4f}",
                'nmi': f"{eval_metrics['nmi']:.4f}"
            })
            
            # Periodic checkpoint
            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(f'epoch_{epoch+1}.pth')
        
        # Final evaluation
        final_metrics = self._evaluate(train_loader, labels)
        
        # Save training history
        self._save_history(training_history)
        
        # Print summary
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best ACC: {self.best_metric:.4f}")
        print(f"Final Metrics: {final_metrics}")
        print("="*60)
        
        return {
            'final': final_metrics,
            'best': best_metrics,
            'history': training_history
        }
    
    def _train_epoch(self, dataloader: DataLoader, ablation_mode: str) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        for batch in dataloader:
            views = [v.to(self.device) for v in batch['views']]
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss, loss_dict = self.model.compute_loss(views, mask, ablation_mode)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0.0) + v
            num_batches += 1
            self.global_step += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        for k in loss_components:
            loss_components[k] /= num_batches
        
        return {'loss': avg_loss, **loss_components}
    
    def _update_clustering(self, dataloader: DataLoader):
        """Update clustering centroids (E-step)"""
        self.model.eval()
        all_embeddings = []
        all_indices = []
        
        with torch.no_grad():
            for batch in dataloader:
                views = [v.to(self.device) for v in batch['views']]
                mask = batch['mask'].to(self.device)
                indices = batch['indices']
                
                outputs = self.model(views, mask)
                all_embeddings.append(outputs['consensus'])
                all_indices.append(indices)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_indices = torch.cat(all_indices, dim=0).numpy()
        
        # 按原始顺序排序
        order = np.argsort(all_indices)
        all_embeddings = all_embeddings[order]
        
        # Update centroids using K-Means
        from sklearn.cluster import KMeans
        embeddings_np = all_embeddings.cpu().numpy()
        kmeans = KMeans(n_clusters=self.model.num_clusters, n_init=10)
        kmeans.fit(embeddings_np)
        
        self.model.clustering.centroids.data = torch.FloatTensor(
            kmeans.cluster_centers_
        ).to(self.device)
    
    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader, labels: np.ndarray) -> Dict:
        """Evaluate clustering performance"""
        self.model.eval()
        all_embeddings = []
        all_predictions = []
        all_indices = []
        
        for batch in dataloader:
            views = [v.to(self.device) for v in batch['views']]
            mask = batch['mask'].to(self.device)
            indices = batch['indices']  # 获取样本索引
            
            outputs = self.model(views, mask)
            all_embeddings.append(outputs['consensus'].cpu())
            all_predictions.append(outputs['assignments'].cpu())
            all_indices.append(indices)
        
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        predictions = torch.cat(all_predictions, dim=0).numpy()
        indices = torch.cat(all_indices, dim=0).numpy()
        
        # 按原始顺序排序，确保与 labels 对应
        order = np.argsort(indices)
        embeddings = embeddings[order]
        predictions = predictions[order]
        
        metrics = evaluate_clustering(labels, predictions, embeddings)
        return metrics
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'global_step': self.global_step
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.experiment_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.experiment_dir / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.global_step = checkpoint['global_step']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _save_history(self, history: List[Dict]):
        """Save training history to JSON"""
        with open(self.experiment_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.should_stop


def run_training(
    config: ExperimentConfig,
    train_loader: DataLoader,
    labels: np.ndarray,
    view_dims: List[int],
    ablation_mode: str = "full"
) -> Dict:
    """
    Run full training pipeline
    
    Args:
        config: Experiment configuration
        train_loader: Training data loader
        labels: Ground truth labels
        view_dims: Dimensions of each view
        ablation_mode: Ablation study mode
    
    Returns:
        Training results
    """
    # Create model
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=config.model.latent_dim,
        hidden_dims=config.model.hidden_dims,
        num_clusters=config.model.num_clusters,
        flow_hidden_dim=config.model.flow_hidden_dim,
        flow_num_layers=config.model.flow_num_layers,
        time_dim=config.model.time_dim,
        ode_steps=config.model.ode_steps,
        sigma_min=config.model.sigma_min,
        kernel_type=config.model.kernel_type,
        kernel_gamma=config.model.kernel_gamma,
        lambda_gw=config.model.lambda_gw,
        lambda_cluster=config.model.lambda_cluster,
        lambda_recon=config.model.lambda_recon,
        lambda_contrastive=config.model.lambda_contrastive,
        dropout=config.model.dropout
    )
    
    # Create experiment directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{config.experiment_name}_{timestamp}"
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config.training,
        experiment_dir=experiment_dir,
        device=config.training.device
    )
    
    # Run training
    results = trainer.train(
        train_loader=train_loader,
        labels=labels,
        ablation_mode=ablation_mode
    )
    
    return results


if __name__ == "__main__":
    # Test training pipeline
    print("Testing training pipeline...")
    
    from .config import get_default_config
    from torch.utils.data import TensorDataset
    
    # Create dummy data
    batch_size = 128
    n_samples = 500
    view_dims = [100, 200, 150]
    num_clusters = 10
    
    views = [torch.randn(n_samples, dim) for dim in view_dims]
    labels = np.random.randint(0, num_clusters, n_samples)
    mask = torch.ones(n_samples, len(view_dims))
    
    # Create dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, views, mask, labels):
            self.views = views
            self.mask = mask
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return {
                'views': [v[idx] for v in self.views],
                'mask': self.mask[idx],
                'label': self.labels[idx]
            }
    
    dataset = DummyDataset(views, mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = OTCFM(
        view_dims=view_dims,
        latent_dim=128,
        num_clusters=num_clusters
    )
    
    # Quick training test
    from .config import TrainingConfig
    config = TrainingConfig()
    config.epochs = 5  # Quick test
    
    trainer = Trainer(
        model=model,
        config=config,
        experiment_dir="test_experiment",
        device='cpu'
    )
    
    results = trainer.train(dataloader, labels)
    print(f"Final ACC: {results['final']['acc']:.4f}")
    print("Training pipeline test passed!")
