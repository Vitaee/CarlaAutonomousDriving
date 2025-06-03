import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from model import NvidiaModelTransferLearning
from dataset_loader import CarlaDataset
from config import config


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        backbone, head = [], []
        for n,p in model.named_parameters():
            (head if n.startswith('regressor') else backbone).append(p)

        self.optimizer = optim.Adam([
            {'params': backbone, 'lr': 1e-4},
            {'params': head,     'lr': 1e-3}
        ], weight_decay=1e-4)
        

       
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.early_stopping_patience
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        with tqdm(self.train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation", leave=False) as pbar:
                for images, targets in pbar:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, val_loss, filepath):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, filepath)
    
    def early_stop_check(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience


def create_data_loaders(batch_size=64, num_workers=24, use_all_cameras=True):
    """Create train and validation data loaders from all CARLA datasets"""
    
    datasets_path = Path("data")
    town_folders = [
        "dataset_carla_001_Town01",
        "dataset_carla_001_Town02", 
        "dataset_carla_001_Town03",
        "dataset_carla_001_Town04",
        "dataset_carla_001_Town05",
        #"dataset_carla_001_Town10HD_Opt"
    ]
    
    all_datasets = []
    all_samplers = []
    
    for town_folder in town_folders:
        town_path = datasets_path / town_folder
        if town_path.exists():
            print(f"Loading dataset: {town_folder}")
            dataset = CarlaDataset(
                root_dir=str(town_path),
                use_all_cameras=use_all_cameras
            )
            all_datasets.append(dataset)
            all_samplers.append(dataset.sampler)
            print(f"  Loaded {len(dataset)} samples")
        else:
            print(f"Warning: {town_folder} not found")
    
    if not all_datasets:
        raise ValueError("No datasets found!")
    
    # Combine all datasets
    combined_dataset = ConcatDataset(all_datasets)
    print(f"Total combined samples: {len(combined_dataset)}")
    
    # Create combined sampler weights
    combined_weights = []
    cumulative_size = 0
    for dataset, sampler in zip(all_datasets, all_samplers):
        # Get weights from the sampler
        dataset_weights = sampler.weights
        combined_weights.extend(dataset_weights)
        cumulative_size += len(dataset)
    
    # Create combined sampler
    combined_sampler = torch.utils.data.WeightedRandomSampler(
        weights=combined_weights,
        num_samples=len(combined_weights),
        replacement=True
    )
    
    # Split into train/val
    train_size = int(config.train_split_size * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create train sampler for the subset
    train_indices = train_dataset.indices
    train_weights = [combined_weights[i] for i in train_indices]
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # Use balanced sampler instead of shuffle=True
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train CARLA Steering Model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use_all_cameras", action="store_true", default=True, 
                       help="Use all three cameras (center, left, right)")
    parser.add_argument("--run_name", type=str, default="carla_steering", 
                       help="Run name for tensorboard")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Update config
    config.learning_rate = args.lr
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_all_cameras=args.use_all_cameras
    )
    
    # Create model
    print("Creating model...")
    model = NvidiaModelTransferLearning()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device)
    
    # Setup tensorboard
    writer = SummaryWriter(f'logs/{args.run_name}')
    
    # Create save directory
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    print(f"\nStarting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Use all cameras: {args.use_all_cameras}")
    
    start_time = time.time()
    epoch = 0
    val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = trainer.train_epoch()
        
        # Validate
        val_loss = trainer.validate_epoch()
        
        # Update scheduler
        trainer.scheduler.step(val_loss)
        
        # Log to tensorboard
        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        
        writer.add_scalar('Learning_Rate', 
                         trainer.optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_loss < trainer.best_val_loss:
            best_model_path = save_dir / f"{args.run_name}_best.pt"
            trainer.save_checkpoint(epoch, val_loss, best_model_path)
            print(f"New best model saved: {val_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = save_dir / f"{args.run_name}_epoch_{epoch}.pt"
            trainer.save_checkpoint(epoch, val_loss, checkpoint_path)
        
        # Early stopping check
        if trainer.early_stop_check(val_loss):
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    final_model_path = save_dir / f"{args.run_name}_final.pt"
    trainer.save_checkpoint(epoch, val_loss, final_model_path)
    
    writer.close()
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")


if __name__ == '__main__':
    main()