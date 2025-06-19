import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast  # Mixed precision training
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging

from model import ProductionCarlaModel, ModelTrainer
from dataset_loader import get_enhanced_dataset, get_enhanced_dataloader, get_training_dataloaders
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedTrainer:
    """Enhanced trainer for ProductionCarlaModel with multi-task learning"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Mixed precision training
        self.scaler = GradScaler() if device.type == 'cuda' else None
        
        # Use the specialized ModelTrainer for loss computation
        self.model_trainer = ModelTrainer(model, device)
        
        # Optimizer with different learning rates for different parts
        backbone_params = []
        fusion_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif any(module in name for module in ['camera_fusion', 'lidar_processor', 'temporal_lstm']):
                fusion_params.append(param)
            else:
                head_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': config.learning_rate * 0.1, 'weight_decay': config.weight_decay * 2},  # Lower LR for backbone
            {'params': fusion_params, 'lr': config.learning_rate * 0.5, 'weight_decay': config.weight_decay},      # Medium LR for fusion
            {'params': head_params, 'lr': config.learning_rate, 'weight_decay': config.weight_decay * 0.5}        # Full LR for heads
        ])
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=config.scheduler_patience, 
            factor=config.scheduler_factor,
            min_lr=config.min_learning_rate
        )
        
        # Warmup scheduler
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.warmup_epochs
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.early_stopping_patience
        
        # Metrics tracking
        self.train_metrics = {'loss': [], 'steering_loss': [], 'speed_loss': [], 'brake_loss': []}
        self.val_metrics = {'loss': [], 'steering_loss': [], 'speed_loss': [], 'brake_loss': []}
        
    def train_epoch(self, epoch):
        """Train for one epoch with multi-task learning and gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        total_steering_loss = 0.0
        total_speed_loss = 0.0
        total_brake_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Gradient accumulation
        accumulation_steps = config.gradient_accumulation_steps
        
        with tqdm(self.train_loader, desc=f"Training Epoch {epoch}", leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                center_imgs = batch['center_imgs'].to(self.device, non_blocking=True)
                left_imgs = batch['left_imgs'].to(self.device, non_blocking=True)
                right_imgs = batch['right_imgs'].to(self.device, non_blocking=True)
                lidar_points = batch['lidar_points'].to(self.device, non_blocking=True)
                
                # Targets - use only the last frame for prediction
                targets = {
                    'steering': batch['steering'][:, -1].to(self.device, non_blocking=True),  # (B,)
                    'speed': batch['speed'][:, -1].to(self.device, non_blocking=True),        # (B,)
                    'emergency_brake': batch['emergency_brake'][:, -1].to(self.device, non_blocking=True)  # (B,)
                }
                
                # Forward pass
                with autocast(enabled=self.scaler is not None):
                    outputs = self.model(
                        center_imgs=center_imgs,
                        left_imgs=left_imgs,
                        right_imgs=right_imgs,
                        lidar_points=lidar_points
                    )
                    
                    # Compute multi-task loss using the specialized trainer
                    loss = self.model_trainer.compute_loss(outputs, targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
                
                # Compute individual losses for monitoring
                with torch.no_grad():
                    steering_loss = nn.MSELoss()(outputs['steering'], targets['steering'])
                    speed_loss = nn.MSELoss()(outputs['speed'], targets['speed'])
                    brake_loss = nn.CrossEntropyLoss()(outputs['emergency_brake'], targets['emergency_brake'])
                
                # Backward pass with mixed precision
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimize every accumulation_steps or at the end of epoch
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    if self.scaler:
                        # Gradient clipping with mixed precision
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.gradient_clip_value)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.gradient_clip_value)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item() * accumulation_steps  # Restore original scale
                total_steering_loss += steering_loss.item()
                total_speed_loss += speed_loss.item()
                total_brake_loss += brake_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'steer': f'{steering_loss.item():.4f}',
                    'speed': f'{speed_loss.item():.4f}',
                    'brake': f'{brake_loss.item():.4f}'
                })
        
        # Update warmup scheduler for first few epochs
        if epoch <= config.warmup_epochs:
            self.warmup_scheduler.step()
        
        avg_metrics = {
            'loss': total_loss / num_batches,
            'steering_loss': total_steering_loss / num_batches,
            'speed_loss': total_speed_loss / num_batches,
            'brake_loss': total_brake_loss / num_batches
        }
        
        return avg_metrics
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_steering_loss = 0.0
        total_speed_loss = 0.0
        total_brake_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Safety metrics
        high_uncertainty_count = 0
        high_anomaly_count = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"Validation Epoch {epoch}", leave=False) as pbar:
                for batch in pbar:
                    # Move data to device
                    center_imgs = batch['center_imgs'].to(self.device, non_blocking=True)
                    left_imgs = batch['left_imgs'].to(self.device, non_blocking=True)
                    right_imgs = batch['right_imgs'].to(self.device, non_blocking=True)
                    lidar_points = batch['lidar_points'].to(self.device, non_blocking=True)
                    
                    # Targets - use only the last frame for prediction
                    targets = {
                        'steering': batch['steering'][:, -1].to(self.device, non_blocking=True),  # (B,)
                        'speed': batch['speed'][:, -1].to(self.device, non_blocking=True),        # (B,)
                        'emergency_brake': batch['emergency_brake'][:, -1].to(self.device, non_blocking=True)  # (B,)
                    }
                    
                    # Forward pass
                    outputs = self.model(
                        center_imgs=center_imgs,
                        left_imgs=left_imgs,
                        right_imgs=right_imgs,
                        lidar_points=lidar_points
                    )
                    
                    # Compute losses
                    loss = self.model_trainer.compute_loss(outputs, targets)
                    steering_loss = nn.MSELoss()(outputs['steering'], targets['steering'])
                    speed_loss = nn.MSELoss()(outputs['speed'], targets['speed'])
                    brake_loss = nn.CrossEntropyLoss()(outputs['emergency_brake'], targets['emergency_brake'])
                    
                    # Safety metrics
                    high_uncertainty_count += (outputs['uncertainty'] > config.uncertainty_threshold).sum().item()
                    high_anomaly_count += (outputs['anomaly_score'] > config.anomaly_threshold).sum().item()
                    
                    # Update metrics
                    total_loss += loss.item()
                    total_steering_loss += steering_loss.item()
                    total_speed_loss += speed_loss.item()
                    total_brake_loss += brake_loss.item()
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'steer': f'{steering_loss.item():.4f}',
                        'speed': f'{speed_loss.item():.4f}',
                        'brake': f'{brake_loss.item():.4f}'
                    })
        
        avg_metrics = {
            'loss': total_loss / num_batches,
            'steering_loss': total_steering_loss / num_batches,
            'speed_loss': total_speed_loss / num_batches,
            'brake_loss': total_brake_loss / num_batches,
            'high_uncertainty_rate': high_uncertainty_count / (num_batches * config.batch_size * config.sequence_length),
            'high_anomaly_rate': high_anomaly_count / (num_batches * config.batch_size * config.sequence_length)
        }
        
        return avg_metrics
    
    def save_checkpoint(self, epoch, val_loss, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': config.__dict__
        }
        torch.save(checkpoint, filepath)
    
    def early_stop_check(self, val_loss):
        """Check for early stopping"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience


def create_enhanced_data_loaders():
    """Create enhanced data loaders for ProductionCarlaModel"""
    
    datasets_path = Path(config.enhanced_data_dir)
    all_datasets = []
    
    for town_folder in config.town_datasets:
        town_path = datasets_path / town_folder
        if town_path.exists():
            logger.info(f"Loading enhanced dataset: {town_folder}")
            dataset = get_enhanced_dataset(
                root_dir=str(town_path),
                sequence_length=config.sequence_length,
                image_size=config.image_size,
                augment=True  # Enable augmentation for training
            )
            all_datasets.append(dataset)
            logger.info(f"  Loaded {len(dataset)} sequences")
        else:
            logger.warning(f"Dataset not found: {town_folder}")
    
    if not all_datasets:
        raise ValueError("No enhanced datasets found!")
    
    # Combine all datasets
    if len(all_datasets) > 1:
        combined_dataset = ConcatDataset(all_datasets)
        logger.info(f"Combined dataset size: {len(combined_dataset)} sequences")
    else:
        combined_dataset = all_datasets[0]
        logger.info(f"Single dataset size: {len(combined_dataset)} sequences")
    
    # Create train/val split
    train_size = int(config.train_split_size * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    logger.info(f"Train sequences: {len(train_dataset)}")
    logger.info(f"Validation sequences: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2  # Prefetch more batches for efficiency
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2  # Prefetch more batches for efficiency
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train ProductionCarlaModel with Enhanced Dataset")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--epochs", type=int, default=config.epochs_count, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--run_name", type=str, default="enhanced_carla_model", help="Run name for tensorboard")
    parser.add_argument("--num_workers", type=int, default=config.num_workers, help="Number of data loader workers")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Update config
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.num_workers = args.num_workers
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating enhanced data loaders...")
    train_loader, val_loader = create_enhanced_data_loaders()
    
    # Create model
    logger.info("Creating ProductionCarlaModel...")
    model = ProductionCarlaModel(
        lidar_input_size=config.max_lidar_points,
        num_history_frames=config.sequence_length
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = EnhancedTrainer(model, train_loader, val_loader, device)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Setup tensorboard
    writer = SummaryWriter(f'{config.log_dir}/{args.run_name}')
    
    # Create save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    logger.info(f"\nStarting training...")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Sequence length: {config.sequence_length}")
    logger.info(f"Image size: {config.image_size}")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(epoch)
        
        # Validate
        val_metrics = trainer.validate_epoch(epoch)
        
        # Update scheduler (after warmup)
        if epoch > config.warmup_epochs:
            trainer.scheduler.step(val_metrics['loss'])
        
        # Log metrics
        writer.add_scalars('Loss/Total', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        
        writer.add_scalars('Loss/Steering', {
            'train': train_metrics['steering_loss'],
            'val': val_metrics['steering_loss']
        }, epoch)
        
        writer.add_scalars('Loss/Speed', {
            'train': train_metrics['speed_loss'],
            'val': val_metrics['speed_loss']
        }, epoch)
        
        writer.add_scalars('Loss/Brake', {
            'train': train_metrics['brake_loss'],
            'val': val_metrics['brake_loss']
        }, epoch)
        
        writer.add_scalar('Safety/High_Uncertainty_Rate', val_metrics['high_uncertainty_rate'], epoch)
        writer.add_scalar('Safety/High_Anomaly_Rate', val_metrics['high_anomaly_rate'], epoch)
        
        # Log learning rates
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            writer.add_scalar(f'Learning_Rate/Group_{i}', param_group['lr'], epoch)
        
        # Print metrics
        logger.info(f"Train - Total: {train_metrics['loss']:.6f}, "
                   f"Steer: {train_metrics['steering_loss']:.6f}, "
                   f"Speed: {train_metrics['speed_loss']:.6f}, "
                   f"Brake: {train_metrics['brake_loss']:.6f}")
        
        logger.info(f"Val   - Total: {val_metrics['loss']:.6f}, "
                   f"Steer: {val_metrics['steering_loss']:.6f}, "
                   f"Speed: {val_metrics['speed_loss']:.6f}, "
                   f"Brake: {val_metrics['brake_loss']:.6f}")
        
        logger.info(f"Safety - Uncertainty: {val_metrics['high_uncertainty_rate']:.3f}, "
                   f"Anomaly: {val_metrics['high_anomaly_rate']:.3f}")
        
        # Save best model
        if val_metrics['loss'] < trainer.best_val_loss:
            best_model_path = save_dir / f"{args.run_name}_best.pt"
            trainer.save_checkpoint(epoch, val_metrics['loss'], best_model_path)
            logger.info(f"New best model saved: {val_metrics['loss']:.6f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = save_dir / f"{args.run_name}_epoch_{epoch}.pt"
            trainer.save_checkpoint(epoch, val_metrics['loss'], checkpoint_path)
        
        # Early stopping check
        if trainer.early_stop_check(val_metrics['loss']):
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    final_model_path = save_dir / f"{args.run_name}_final.pt"
    trainer.save_checkpoint(epoch, val_metrics['loss'], final_model_path)
    
    writer.close()
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nTraining completed in {elapsed_time:.2f} seconds")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")


if __name__ == '__main__':
    main()