import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from model import MultiControlAutonomousModel, SpeedAwareAutonomousModel
from dataset_loader import create_multi_output_data_loaders
from config import config


class MultiControlTrainer:
    def __init__(self, model, train_loader, val_loader, device, use_speed_input=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_speed_input = use_speed_input
        
        # Different loss functions for different outputs
        self.steering_criterion = nn.MSELoss()  # Precise steering control
        self.throttle_criterion = nn.MSELoss()  # Smooth acceleration
        self.brake_criterion = nn.MSELoss()     # Smooth braking
        
        # Loss weights (can be tuned)
        self.loss_weights = {
            'steering': 2.0,  # Most important for safety
            'throttle': 1.0,
            'brake': 1.5      # Important for safety
        }
        
        # Optimizer with different learning rates for different components
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'conv_layers' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': 1e-4},
            {'params': head_params, 'lr': 1e-3}
        ], weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 10
        
    def train_epoch(self):
        self.model.train()
        total_losses = {'steering': 0.0, 'throttle': 0.0, 'brake': 0.0, 'total': 0.0}
        
        with tqdm(self.train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, batch_data in enumerate(pbar):
                
                if self.use_speed_input:
                    images, targets, speeds = batch_data
                    images = images.to(self.device, non_blocking=True)
                    speeds = speeds.to(self.device, non_blocking=True)
                else:
                    images, targets = batch_data
                    images = images.to(self.device, non_blocking=True)
                    speeds = None
                
                # Move targets to device
                steering_targets = targets['steering'].to(self.device, non_blocking=True)
                throttle_targets = targets['throttle'].to(self.device, non_blocking=True)
                brake_targets = targets['brake'].to(self.device, non_blocking=True)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if self.use_speed_input:
                    outputs = self.model(images, speeds)
                else:
                    outputs = self.model(images)
                
                # Calculate individual losses
                steering_loss = self.steering_criterion(outputs['steering'], steering_targets)
                throttle_loss = self.throttle_criterion(outputs['throttle'], throttle_targets)
                brake_loss = self.brake_criterion(outputs['brake'], brake_targets)
                
                # Weighted total loss
                total_loss = (
                    self.loss_weights['steering'] * steering_loss +
                    self.loss_weights['throttle'] * throttle_loss +
                    self.loss_weights['brake'] * brake_loss
                )
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Track losses
                total_losses['steering'] += steering_loss.item()
                total_losses['throttle'] += throttle_loss.item()
                total_losses['brake'] += brake_loss.item()
                total_losses['total'] += total_loss.item()
                
                pbar.set_postfix({
                    'total': f'{total_loss.item():.4f}',
                    'steer': f'{steering_loss.item():.4f}',
                    'throttle': f'{throttle_loss.item():.4f}',
                    'brake': f'{brake_loss.item():.4f}'
                })
        
        # Return average losses
        n_batches = len(self.train_loader)
        return {key: loss / n_batches for key, loss in total_losses.items()}
    
    def validate_epoch(self):
        self.model.eval()
        total_losses = {'steering': 0.0, 'throttle': 0.0, 'brake': 0.0, 'total': 0.0}
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation", leave=False) as pbar:
                for batch_data in pbar:
                    
                    if self.use_speed_input:
                        images, targets, speeds = batch_data
                        images = images.to(self.device, non_blocking=True)
                        speeds = speeds.to(self.device, non_blocking=True)
                    else:
                        images, targets = batch_data
                        images = images.to(self.device, non_blocking=True)
                        speeds = None
                    
                    steering_targets = targets['steering'].to(self.device, non_blocking=True)
                    throttle_targets = targets['throttle'].to(self.device, non_blocking=True)
                    brake_targets = targets['brake'].to(self.device, non_blocking=True)
                    
                    # Forward pass
                    if self.use_speed_input:
                        outputs = self.model(images, speeds)
                    else:
                        outputs = self.model(images)
                    
                    # Calculate losses
                    steering_loss = self.steering_criterion(outputs['steering'], steering_targets)
                    throttle_loss = self.throttle_criterion(outputs['throttle'], throttle_targets)
                    brake_loss = self.brake_criterion(outputs['brake'], brake_targets)
                    
                    total_loss = (
                        self.loss_weights['steering'] * steering_loss +
                        self.loss_weights['throttle'] * throttle_loss +
                        self.loss_weights['brake'] * brake_loss
                    )
                    
                    total_losses['steering'] += steering_loss.item()
                    total_losses['throttle'] += throttle_loss.item()
                    total_losses['brake'] += brake_loss.item()
                    total_losses['total'] += total_loss.item()
                    
                    pbar.set_postfix({
                        'total': f'{total_loss.item():.4f}',
                        'steer': f'{steering_loss.item():.4f}',
                        'throttle': f'{throttle_loss.item():.4f}',
                        'brake': f'{brake_loss.item():.4f}'
                    })
        
        n_batches = len(self.val_loader)
        return {key: loss / n_batches for key, loss in total_losses.items()}
    
    def save_checkpoint(self, epoch, val_losses, filepath):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_losses': val_losses,
            'loss_weights': self.loss_weights,
            'use_speed_input': self.use_speed_input
        }
        torch.save(checkpoint, filepath)


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Control CARLA Model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use_all_cameras", action="store_true", default=True)
    parser.add_argument("--use_speed_input", action="store_true", 
                       help="Use current speed as additional input")
    parser.add_argument("--model_type", choices=['multi_control', 'speed_aware'], 
                       default='multi_control', help="Model architecture to use")
    parser.add_argument("--run_name", type=str, default="carla_multi_control")
    parser.add_argument("--num_workers", type=int, default=24)
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating multi-output data loaders...")
    train_loader, val_loader = create_multi_output_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_all_cameras=args.use_all_cameras,
        use_speed_input=args.use_speed_input
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    if args.model_type == 'multi_control':
        model = MultiControlAutonomousModel(
            pretrained=True, 
            freeze_features=False,
            use_speed_input=args.use_speed_input
        )
    else:  # speed_aware
        model = SpeedAwareAutonomousModel(pretrained=True, freeze_features=False)
        args.use_speed_input = True  # Force speed input for this model
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = MultiControlTrainer(model, train_loader, val_loader, device, args.use_speed_input)
    
    # Setup tensorboard
    writer = SummaryWriter(f'logs/{args.run_name}')
    
    # Create save directory
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n🚗 Starting Multi-Control Autonomous Driving Training...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model type: {args.model_type}")
    print(f"Use speed input: {args.use_speed_input}")
    print(f"Outputs: Steering + Throttle + Brake")
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_losses = trainer.train_epoch()
        
        # Validate  
        val_losses = trainer.validate_epoch()
        
        # Update scheduler
        trainer.scheduler.step(val_losses['total'])
        
        # Log to tensorboard
        for loss_type in ['steering', 'throttle', 'brake', 'total']:
            writer.add_scalars(f'Loss/{loss_type}', {
                'train': train_losses[loss_type],
                'val': val_losses[loss_type]
            }, epoch)
        
        writer.add_scalar('Learning_Rate', 
                         trainer.optimizer.param_groups[0]['lr'], epoch)
        
        # Print results
        print(f"Train - Total: {train_losses['total']:.4f}, "
              f"Steer: {train_losses['steering']:.4f}, "
              f"Throttle: {train_losses['throttle']:.4f}, "
              f"Brake: {train_losses['brake']:.4f}")
        print(f"Val   - Total: {val_losses['total']:.4f}, "
              f"Steer: {val_losses['steering']:.4f}, "
              f"Throttle: {val_losses['throttle']:.4f}, "
              f"Brake: {val_losses['brake']:.4f}")
        print(f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_losses['total'] < trainer.best_val_loss:
            trainer.best_val_loss = val_losses['total']
            best_model_path = save_dir / f"{args.run_name}_best.pt"
            trainer.save_checkpoint(epoch, val_losses, best_model_path)
            print(f"🏆 New best model saved: {val_losses['total']:.4f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = save_dir / f"{args.run_name}_epoch_{epoch}.pt"
            trainer.save_checkpoint(epoch, val_losses, checkpoint_path)
        
        # Early stopping check
        if val_losses['total'] < trainer.best_val_loss:
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1
            
        if trainer.patience_counter >= trainer.patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    final_model_path = save_dir / f"{args.run_name}_final.pt"
    trainer.save_checkpoint(epoch, val_losses, final_model_path)
    
    writer.close()
    
    elapsed_time = time.time() - start_time
    print(f"\n🏁 Multi-Control Training completed in {elapsed_time:.2f} seconds")
    print(f"🏆 Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"📁 Models saved to: {save_dir}")


if __name__ == '__main__':
    main()
    # python train.py --batch_size 128 --epochs 50 --use_all_cameras