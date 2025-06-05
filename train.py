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
        
        # Enhanced loss functions for different outputs
        self.steering_criterion = nn.MSELoss()  # Precise steering control
        self.throttle_criterion = nn.SmoothL1Loss()  # More robust for throttle
        self.brake_criterion = nn.SmoothL1Loss()     # More robust for brake
        
        # Adaptive loss weights based on data characteristics
        self.loss_weights = {
            'steering': 3.0,  # Most important for safety
            'throttle': 2.0,  # Increased weight for better learning
            'brake': 2.5      # Critical for safety
        }
        
        # Loss weight scheduling
        self.initial_weights = self.loss_weights.copy()
        self.weight_schedule = {
            'warmup_epochs': 5,
            'throttle_boost_epochs': [10, 20],  # Boost throttle learning
            'brake_boost_epochs': [15, 25]      # Boost brake learning
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
            {'params': backbone_params, 'lr': 5e-5},  # Lower for pretrained backbone
            {'params': head_params, 'lr': 2e-3}       # Higher for control heads
        ], weight_decay=1e-4)
        
        # Learning rate scheduler with more patience
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=7, factor=0.6, verbose=True
        )
        
        # Early stopping with more patience for multi-control
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 15  # Increased patience
        
        # Training metrics tracking
        self.epoch_metrics = {
            'steering_improvement': [],
            'throttle_improvement': [],
            'brake_improvement': []
        }
        
        print("🚗 Enhanced Multi-Control Trainer Initialized")
        print(f"  🎯 Loss functions: MSE(steering), SmoothL1(throttle/brake)")
        print(f"  ⚖️ Initial weights: {self.loss_weights}")
        print(f"  🧠 Backbone LR: {5e-5:.0e}, Heads LR: {2e-3:.0e}")
    
    def update_loss_weights(self, epoch):
        """Dynamically update loss weights based on training progress"""
        
        # Warmup phase - reduce steering dominance
        if epoch <= self.weight_schedule['warmup_epochs']:
            self.loss_weights['steering'] = 2.0
            self.loss_weights['throttle'] = 1.5
            self.loss_weights['brake'] = 2.0
            
        # Throttle boost phases
        elif epoch in range(self.weight_schedule['throttle_boost_epochs'][0], 
                          self.weight_schedule['throttle_boost_epochs'][1]):
            self.loss_weights['throttle'] = 3.0  # Boost throttle learning
            
        # Brake boost phases  
        elif epoch in range(self.weight_schedule['brake_boost_epochs'][0],
                          self.weight_schedule['brake_boost_epochs'][1]):
            self.loss_weights['brake'] = 3.5   # Boost brake learning
            
        # Final balanced phase
        else:
            self.loss_weights['steering'] = 2.5
            self.loss_weights['throttle'] = 2.0
            self.loss_weights['brake'] = 2.5
    
    def calculate_adaptive_loss(self, outputs, targets, epoch):
        """Calculate adaptive loss with focus balancing"""
        
        # Basic losses
        steering_loss = self.steering_criterion(outputs['steering'], targets['steering'])
        throttle_loss = self.throttle_criterion(outputs['throttle'], targets['throttle'])
        brake_loss = self.brake_criterion(outputs['brake'], targets['brake'])
        
        # Adaptive weighting based on individual performance
        # If throttle/brake are performing poorly, increase their weights
        if hasattr(self, 'last_val_losses'):
            if self.last_val_losses.get('throttle', 1.0) > 0.5:  # Poor throttle performance
                throttle_weight = self.loss_weights['throttle'] * 1.5
            else:
                throttle_weight = self.loss_weights['throttle']
                
            if self.last_val_losses.get('brake', 1.0) > 0.3:  # Poor brake performance
                brake_weight = self.loss_weights['brake'] * 1.5
            else:
                brake_weight = self.loss_weights['brake']
        else:
            throttle_weight = self.loss_weights['throttle']
            brake_weight = self.loss_weights['brake']
        
        # Calculate weighted total loss
        total_loss = (
            self.loss_weights['steering'] * steering_loss +
            throttle_weight * throttle_loss +
            brake_weight * brake_loss
        )
        
        return {
            'steering': steering_loss,
            'throttle': throttle_loss, 
            'brake': brake_loss,
            'total': total_loss
        }
    
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
                
                # Calculate adaptive losses
                targets_dict = {
                    'steering': steering_targets,
                    'throttle': throttle_targets,
                    'brake': brake_targets
                }
                
                # Use current epoch from trainer state
                current_epoch = getattr(self, 'current_epoch', 1)
                losses = self.calculate_adaptive_loss(outputs, targets_dict, current_epoch)
                
                # Backward pass
                losses['total'].backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Track losses
                for key in total_losses:
                    total_losses[key] += losses[key].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'total': f'{losses["total"].item():.4f}',
                    'steer': f'{losses["steering"].item():.4f}',
                    'throttle': f'{losses["throttle"].item():.4f}',
                    'brake': f'{losses["brake"].item():.4f}',
                    'weights': f'S:{self.loss_weights["steering"]:.1f}/T:{throttle_weight if "throttle_weight" in locals() else self.loss_weights["throttle"]:.1f}/B:{brake_weight if "brake_weight" in locals() else self.loss_weights["brake"]:.1f}'
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
                    
                    # Calculate losses (without adaptive weighting for validation)
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
        val_losses = {key: loss / n_batches for key, loss in total_losses.items()}
        
        # Store for adaptive weighting
        self.last_val_losses = val_losses
        
        return val_losses
    
    def save_checkpoint(self, epoch, val_losses, filepath):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_losses': val_losses,
            'loss_weights': self.loss_weights,
            'use_speed_input': self.use_speed_input,
            'epoch_metrics': self.epoch_metrics
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
    parser.add_argument("--run_name", type=str, default="enhanced_multi_control")
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--enable_augmentation", action="store_true", default=True,
                       help="Enable control augmentation for better throttle/brake training")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create enhanced data loaders
    print("Creating ENHANCED multi-output data loaders...")
    train_loader, val_loader = create_multi_output_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_all_cameras=args.use_all_cameras,
        use_speed_input=args.use_speed_input,
        enable_control_augmentation=args.enable_augmentation
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
    
    # Create enhanced trainer
    trainer = MultiControlTrainer(model, train_loader, val_loader, device, args.use_speed_input)
    
    # Setup tensorboard
    writer = SummaryWriter(f'logs/{args.run_name}')
    
    # Create save directory
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n🚗 Starting ENHANCED Multi-Control Training...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model type: {args.model_type}")
    print(f"Use speed input: {args.use_speed_input}")
    print(f"Control augmentation: {args.enable_augmentation}")
    print(f"Outputs: Steering + Throttle + Brake")
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Update trainer state and loss weights
        trainer.current_epoch = epoch
        trainer.update_loss_weights(epoch)
        
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
        
        # Log loss weights
        for control_type, weight in trainer.loss_weights.items():
            writer.add_scalar(f'Weights/{control_type}', weight, epoch)
        
        writer.add_scalar('Learning_Rate', 
                         trainer.optimizer.param_groups[0]['lr'], epoch)
        
        # Print enhanced results
        print(f"📊 Loss Weights: Steering={trainer.loss_weights['steering']:.1f}, "
              f"Throttle={trainer.loss_weights['throttle']:.1f}, "
              f"Brake={trainer.loss_weights['brake']:.1f}")
        print(f"Train - Total: {train_losses['total']:.4f}, "
              f"Steer: {train_losses['steering']:.4f}, "
              f"Throttle: {train_losses['throttle']:.4f}, "
              f"Brake: {train_losses['brake']:.4f}")
        print(f"Val   - Total: {val_losses['total']:.4f}, "
              f"Steer: {val_losses['steering']:.4f}, "
              f"Throttle: {val_losses['throttle']:.4f}, "
              f"Brake: {val_losses['brake']:.4f}")
        print(f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # Enhanced improvement tracking
        if epoch > 1:
            for control in ['steering', 'throttle', 'brake']:
                improvement = trainer.epoch_metrics[f'{control}_improvement']
                if len(improvement) > 0:
                    last_val = improvement[-1]
                    current_val = val_losses[control]
                    improvement_pct = (last_val - current_val) / last_val * 100
                    print(f"  {control.title()} improvement: {improvement_pct:+.2f}%")
                trainer.epoch_metrics[f'{control}_improvement'].append(val_losses[control])
        
        # Save best model
        if val_losses['total'] < trainer.best_val_loss:
            trainer.best_val_loss = val_losses['total']
            best_model_path = save_dir / f"{args.run_name}_best.pt"
            trainer.save_checkpoint(epoch, val_losses, best_model_path)
            print(f"🏆 New best model saved: {val_losses['total']:.4f}")
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = save_dir / f"{args.run_name}_epoch_{epoch}.pt"
            trainer.save_checkpoint(epoch, val_losses, checkpoint_path)
        
        # Early stopping check
        if trainer.patience_counter >= trainer.patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    final_model_path = save_dir / f"{args.run_name}_final.pt"
    trainer.save_checkpoint(epoch, val_losses, final_model_path)
    
    writer.close()
    
    elapsed_time = time.time() - start_time
    print(f"\n🏁 ENHANCED Multi-Control Training completed in {elapsed_time:.2f} seconds")
    print(f"🏆 Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"📁 Models saved to: {save_dir}")
    
    # Final improvement summary
    print(f"\n📈 Final Training Summary:")
    for control in ['steering', 'throttle', 'brake']:
        improvements = trainer.epoch_metrics[f'{control}_improvement']
        if len(improvements) > 1:
            total_improvement = (improvements[0] - improvements[-1]) / improvements[0] * 100
            print(f"  {control.title()}: {total_improvement:+.2f}% total improvement")


if __name__ == '__main__':
    main()
    # python train.py --batch_size 128 --epochs 50 --use_all_cameras