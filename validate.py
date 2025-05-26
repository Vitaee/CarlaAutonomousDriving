import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dataset_loader
import numpy as np
import pandas as pd
from model import NvidiaMultiOutputModel  # Updated import
from config import config
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm


class MultiOutputValidator:
    """Comprehensive validation class for multi-output autonomous driving model"""
    
    def __init__(self, model_path, device=None):
        self.device = device or config.device
        self.model = self._load_model(model_path)
        self.loss_functions = {
            'steering': nn.MSELoss(),
            'throttle': nn.MSELoss(),
            'brake': nn.MSELoss()
        }
        
    def _load_model(self, model_path):
        """Load the trained multi-output model"""
        model = NvidiaMultiOutputModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        model.to(self.device)
        model.eval()
        return model
    
    def validate_comprehensive(self, val_loader, save_results=True, output_dir="./validation_results"):
        """
        Comprehensive validation with detailed metrics and visualizations
        """
        print("Starting comprehensive validation...")
        
        # Create output directory
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        # Storage for all predictions and targets
        all_predictions = {
            'steering': [],
            'throttle': [],
            'brake': []
        }
        all_targets = {
            'steering': [],
            'throttle': [],
            'brake': []
        }
        
        # Storage for batch losses
        batch_losses = {
            'steering': [],
            'throttle': [],
            'brake': [],
            'total': []
        }
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(val_loader, desc="Validating")):
                # Move data to device
                data = data.to(self.device)
                targets_device = {}
                for key, value in targets.items():
                    targets_device[key] = value.to(self.device)
                
                # Forward pass
                predictions = self.model(data)
                
                # Calculate losses
                steering_loss = self.loss_functions['steering'](
                    predictions['steering'].float(), 
                    targets_device['steering'].float()
                )
                throttle_loss = self.loss_functions['throttle'](
                    predictions['throttle'].float(), 
                    targets_device['throttle'].float()
                )
                brake_loss = self.loss_functions['brake'](
                    predictions['brake'].float(), 
                    targets_device['brake'].float()
                )
                
                # Weighted total loss
                total_loss = (
                    config.steering_weight * steering_loss + 
                    config.throttle_weight * throttle_loss + 
                    config.brake_weight * brake_loss
                )
                
                # Store batch losses
                batch_losses['steering'].append(steering_loss.item())
                batch_losses['throttle'].append(throttle_loss.item())
                batch_losses['brake'].append(brake_loss.item())
                batch_losses['total'].append(total_loss.item())
                
                # Store predictions and targets for detailed analysis
                for key in ['steering', 'throttle', 'brake']:
                    all_predictions[key].extend(predictions[key].cpu().numpy().tolist())
                    all_targets[key].extend(targets[key].cpu().numpy().tolist())
                
                total_samples += data.size(0)
                
                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    current_losses = {k: np.mean(v) for k, v in batch_losses.items()}
                    print(f"Batch {batch_idx}: "
                          f"Steering: {current_losses['steering']:.6f}, "
                          f"Throttle: {current_losses['throttle']:.6f}, "
                          f"Brake: {current_losses['brake']:.6f}, "
                          f"Total: {current_losses['total']:.6f}")
        
        # Calculate final metrics
        final_metrics = self._calculate_detailed_metrics(all_predictions, all_targets)
        final_losses = {k: np.mean(v) for k, v in batch_losses.items()}
        
        # Print comprehensive results
        self._print_validation_results(final_metrics, final_losses, total_samples)
        
        if save_results:
            # Save results to files
            self._save_validation_results(
                final_metrics, final_losses, batch_losses, 
                all_predictions, all_targets, output_dir
            )
            
            # Create visualizations
            self._create_visualizations(all_predictions, all_targets, output_dir)
        
        return final_metrics, final_losses
    
    def _calculate_detailed_metrics(self, predictions, targets):
        """Calculate comprehensive metrics for each output"""
        metrics = {}
        
        for output_name in ['steering', 'throttle', 'brake']:
            pred = np.array(predictions[output_name])
            target = np.array(targets[output_name])
            
            metrics[output_name] = {
                'mse': mean_squared_error(target, pred),
                'rmse': np.sqrt(mean_squared_error(target, pred)),
                'mae': mean_absolute_error(target, pred),
                'r2': r2_score(target, pred),
                'mean_error': np.mean(pred - target),
                'std_error': np.std(pred - target),
                'max_error': np.max(np.abs(pred - target)),
                'accuracy_95': np.percentile(np.abs(pred - target), 95),
                'prediction_range': (np.min(pred), np.max(pred)),
                'target_range': (np.min(target), np.max(target))
            }
        
        return metrics
    
    def _print_validation_results(self, metrics, losses, total_samples):
        """Print comprehensive validation results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION RESULTS")
        print("="*80)
        print(f"Total Samples Validated: {total_samples}")
        print(f"Device Used: {self.device}")
        
        # Loss Summary
        print(f"\n📊 LOSS SUMMARY:")
        for output_name, loss_value in losses.items():
            print(f"  {output_name.title():>10}: {loss_value:.8f}")
        
        # Detailed Metrics for Each Output
        for output_name in ['steering', 'throttle', 'brake']:
            print(f"\n🎯 {output_name.upper()} CONTROL METRICS:")
            m = metrics[output_name]
            print(f"  MSE:           {m['mse']:.8f}")
            print(f"  RMSE:          {m['rmse']:.8f}")
            print(f"  MAE:           {m['mae']:.8f}")
            print(f"  R² Score:      {m['r2']:.6f}")
            print(f"  Mean Error:    {m['mean_error']:+.6f}")
            print(f"  Std Error:     {m['std_error']:.6f}")
            print(f"  Max Error:     {m['max_error']:.6f}")
            print(f"  95% Accuracy:  {m['accuracy_95']:.6f}")
            print(f"  Pred Range:    ({m['prediction_range'][0]:.3f}, {m['prediction_range'][1]:.3f})")
            print(f"  Target Range:  ({m['target_range'][0]:.3f}, {m['target_range'][1]:.3f})")
        
        # Performance Assessment
        print(f"\n✅ PERFORMANCE ASSESSMENT:")
        steering_quality = "Excellent" if metrics['steering']['rmse'] < 0.05 else "Good" if metrics['steering']['rmse'] < 0.1 else "Needs Improvement"
        throttle_quality = "Excellent" if metrics['throttle']['rmse'] < 0.15 else "Good" if metrics['throttle']['rmse'] < 0.25 else "Needs Improvement"
        brake_quality = "Excellent" if metrics['brake']['rmse'] < 0.15 else "Good" if metrics['brake']['rmse'] < 0.25 else "Needs Improvement"
        
        print(f"  Steering Control: {steering_quality}")
        print(f"  Throttle Control: {throttle_quality}")
        print(f"  Brake Control:    {brake_quality}")
        print("="*80)
    
    def _save_validation_results(self, metrics, losses, batch_losses, predictions, targets, output_dir):
        """Save validation results to files"""
        
        # Save summary metrics
        summary_data = []
        for output_name in ['steering', 'throttle', 'brake']:
            m = metrics[output_name]
            summary_data.append({
                'output': output_name,
                'loss': losses[output_name],
                'mse': m['mse'],
                'rmse': m['rmse'],
                'mae': m['mae'],
                'r2': m['r2'],
                'mean_error': m['mean_error'],
                'std_error': m['std_error'],
                'max_error': m['max_error'],
                'accuracy_95': m['accuracy_95']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, "validation_summary.csv"), index=False)
        
        # Save batch losses
        batch_losses_df = pd.DataFrame(batch_losses)
        batch_losses_df.to_csv(os.path.join(output_dir, "batch_losses.csv"), index=False)
        
        # Save detailed predictions vs targets
        detailed_data = []
        for i in range(len(predictions['steering'])):
            detailed_data.append({
                'sample_idx': i,
                'steering_pred': predictions['steering'][i],
                'steering_target': targets['steering'][i],
                'steering_error': predictions['steering'][i] - targets['steering'][i],
                'throttle_pred': predictions['throttle'][i],
                'throttle_target': targets['throttle'][i],
                'throttle_error': predictions['throttle'][i] - targets['throttle'][i],
                'brake_pred': predictions['brake'][i],
                'brake_target': targets['brake'][i],
                'brake_error': predictions['brake'][i] - targets['brake'][i],
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(os.path.join(output_dir, "detailed_predictions.csv"), index=False)
        
        print(f"✅ Validation results saved to {output_dir}")
    
    def _create_visualizations(self, predictions, targets, output_dir):
        """Create comprehensive visualizations"""
        
        # 1. Prediction vs Target Scatter Plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, output_name in enumerate(['steering', 'throttle', 'brake']):
            pred = np.array(predictions[output_name])
            target = np.array(targets[output_name])
            
            axes[idx].scatter(target, pred, alpha=0.6, s=20)
            min_val = min(target.min(), pred.min())
            max_val = max(target.max(), pred.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[idx].set_xlabel(f'Actual {output_name.title()}')
            axes[idx].set_ylabel(f'Predicted {output_name.title()}')
            axes[idx].set_title(f'{output_name.title()} Control: Predicted vs Actual')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prediction_scatter_plots.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Error Distribution Histograms
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, output_name in enumerate(['steering', 'throttle', 'brake']):
            pred = np.array(predictions[output_name])
            target = np.array(targets[output_name])
            errors = pred - target
            
            axes[idx].hist(errors, bins=50, alpha=0.7, edgecolor='black')
            axes[idx].axvline(np.mean(errors), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(errors):.4f}')
            axes[idx].axvline(np.median(errors), color='green', linestyle='--', 
                             label=f'Median: {np.median(errors):.4f}')
            axes[idx].set_xlabel(f'{output_name.title()} Prediction Error')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{output_name.title()} Control: Error Distribution')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Time Series Plot (first 1000 samples)
        n_samples = min(1000, len(predictions['steering']))
        x = np.arange(n_samples)
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        for idx, output_name in enumerate(['steering', 'throttle', 'brake']):
            pred = np.array(predictions[output_name][:n_samples])
            target = np.array(targets[output_name][:n_samples])
            
            axes[idx].plot(x, target, label='Actual', color='blue', linewidth=1.5, alpha=0.8)
            axes[idx].plot(x, pred, label='Predicted', color='red', linewidth=1.5, alpha=0.8)
            axes[idx].set_ylabel(f'{output_name.title()}')
            axes[idx].set_title(f'{output_name.title()} Control: Time Series Comparison')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Sample Index')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "time_series_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {output_dir}")


def main():
    """Main validation function"""
    print("🚗 Multi-Output Autonomous Driving Model Validation")
    print("="*60)
    
    # Load validation dataset
    _, val_subset_loader = dataset_loader.get_data_subsets_loaders(
        dataset_types=['carla_001'], 
        train_only_center=False,
        batch_size=32  # Use smaller batch size for validation
    )
    
    print(f"Validation dataset size: {len(val_subset_loader.dataset)} samples")
    
    # Initialize validator
    validator = MultiOutputValidator(
        model_path="./save_new/model.pt",  # Update path as needed
        device=config.device
    )
    
    # Run comprehensive validation
    try:
        metrics, losses = validator.validate_comprehensive(
            val_loader=val_subset_loader,
            save_results=True,
            output_dir="./validation_results"
        )
        
        print("\n✅ Validation completed successfully!")
        print("📁 Results saved to './validation_results' directory")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise


if __name__ == '__main__':
    main()