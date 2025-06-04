import cv2
import numpy as np
import torch
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from collections import defaultdict
import concurrent.futures
import threading
from queue import Queue
import multiprocessing as mp

from config import config
from model import MultiControlAutonomousModel
from dataset_loader import get_inference_dataset


class OptimizedDataLoader:
    """High-performance dataloader optimized for your hardware"""
    
    def __init__(self, dataset, batch_size=256, num_workers=28, prefetch_factor=4):
        """
        Optimized for your 32-core CPU:
        - Use 28 workers (leave 4 cores for main process + OS)
        - Large batch size to utilize 64GB RAM
        - High prefetch factor for continuous data flow
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True,  # Keep workers alive
            drop_last=False
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


class FastMultiControlMetrics:
    """Vectorized metrics calculator for multi-control outputs (steering, throttle, brake)"""
    
    def __init__(self):
        # Store predictions and targets for each control type
        self.steering_predictions = []
        self.steering_targets = []
        self.throttle_predictions = []
        self.throttle_targets = []
        self.brake_predictions = []
        self.brake_targets = []
        
        self.inference_times = []
        self.batch_sizes = []
        
    def add_batch_predictions(self, predictions: Dict, targets: Dict, inference_time: float):
        """Add batch predictions for all three control outputs"""
        # Convert tensors to numpy if needed
        if isinstance(predictions['steering'], torch.Tensor):
            steer_pred = predictions['steering'].cpu().numpy()
            throttle_pred = predictions['throttle'].cpu().numpy()
            brake_pred = predictions['brake'].cpu().numpy()
        else:
            steer_pred = predictions['steering']
            throttle_pred = predictions['throttle']
            brake_pred = predictions['brake']
            
        if isinstance(targets['steering'], torch.Tensor):
            steer_target = targets['steering'].cpu().numpy()
            throttle_target = targets['throttle'].cpu().numpy()
            brake_target = targets['brake'].cpu().numpy()
        else:
            steer_target = targets['steering']
            throttle_target = targets['throttle']
            brake_target = targets['brake']
        
        # Store predictions and targets
        self.steering_predictions.extend(steer_pred.tolist())
        self.steering_targets.extend(steer_target.tolist())
        self.throttle_predictions.extend(throttle_pred.tolist())
        self.throttle_targets.extend(throttle_target.tolist())
        self.brake_predictions.extend(brake_pred.tolist())
        self.brake_targets.extend(brake_target.tolist())
        
        # Store timing info
        batch_size = len(steer_pred)
        time_per_sample = inference_time / batch_size
        self.inference_times.extend([time_per_sample] * batch_size)
        self.batch_sizes.append(batch_size)
        
    def calculate_metrics_vectorized(self) -> Dict:
        """Calculate comprehensive metrics for all three control outputs"""
        if not self.steering_predictions:
            return {}
            
        # Convert to numpy arrays
        steer_pred = np.array(self.steering_predictions, dtype=np.float32)
        steer_target = np.array(self.steering_targets, dtype=np.float32)
        throttle_pred = np.array(self.throttle_predictions, dtype=np.float32)
        throttle_target = np.array(self.throttle_targets, dtype=np.float32)
        brake_pred = np.array(self.brake_predictions, dtype=np.float32)
        brake_target = np.array(self.brake_targets, dtype=np.float32)
        
        # Calculate metrics for each control output
        steering_metrics = self._calculate_control_metrics(steer_pred, steer_target, "steering", use_degrees=True)
        throttle_metrics = self._calculate_control_metrics(throttle_pred, throttle_target, "throttle", use_degrees=False)
        brake_metrics = self._calculate_control_metrics(brake_pred, brake_target, "brake", use_degrees=False)
        
        # Calculate autonomous driving specific metrics (mainly for steering)
        direction_accuracy = self._calculate_direction_accuracy_vectorized(steer_pred, steer_target)
        angle_accuracies = self._calculate_angle_accuracies_vectorized(steer_pred, steer_target)
        
        # Performance metrics
        avg_inference_time = np.mean(self.inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        # Combined metrics
        combined_metrics = {
            # Steering metrics (convert to degrees for readability)
            **{f"steering_{k}": (np.degrees(v) if k in ['mae', 'mse', 'rmse'] else v) 
               for k, v in steering_metrics.items()},
            
            # Throttle and brake metrics (already in 0-1 range)
            **{f"throttle_{k}": v for k, v in throttle_metrics.items()},
            **{f"brake_{k}": v for k, v in brake_metrics.items()},
            
            # Autonomous driving metrics
            'direction_accuracy': direction_accuracy,
            'angle_accuracy_5deg': angle_accuracies['5deg'],
            'angle_accuracy_10deg': angle_accuracies['10deg'],
            'angle_accuracy_15deg': angle_accuracies['15deg'],
            
            # Performance metrics
            'avg_inference_time_ms': avg_inference_time * 1000,
            'fps': fps,
            'num_samples': len(self.steering_predictions),
            'total_batches': len(self.batch_sizes),
            'avg_batch_size': np.mean(self.batch_sizes),
            
            # Combined control quality score
            'overall_control_score': self._calculate_overall_score(steering_metrics, throttle_metrics, brake_metrics)
        }
        
        return combined_metrics
    
    def _calculate_control_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                                 control_name: str, use_degrees: bool = False) -> Dict:
        """Calculate metrics for a single control output"""
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        # Convert to degrees if needed (for steering)
        if use_degrees:
            abs_errors_display = np.degrees(abs_errors)
            errors_display = np.degrees(errors)
        else:
            abs_errors_display = abs_errors
            errors_display = errors
        
        mae = np.mean(abs_errors)
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        
        # Avoid division by zero for R² calculation
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((targets - np.mean(targets))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        
        # MAPE calculation with protection against division by zero
        mape = np.mean(abs_errors / (np.abs(targets) + 1e-8)) * 100
        
        # Percentile errors
        error_percentiles = np.percentile(abs_errors_display, [50, 75, 90, 95, 99])
        
        return {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape,
            'median_abs_error': error_percentiles[0],
            'p75_abs_error': error_percentiles[1],
            'p90_abs_error': error_percentiles[2],
            'p95_abs_error': error_percentiles[3],
            'p99_abs_error': error_percentiles[4],
            'prediction_std': np.std(predictions),
            'target_std': np.std(targets),
            'prediction_range': (float(np.min(predictions)), float(np.max(predictions))),
            'target_range': (float(np.min(targets)), float(np.max(targets)))
        }
    
    def _calculate_direction_accuracy_vectorized(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate steering direction accuracy"""
        straight_threshold = 0.05
        pred_directions = np.where(predictions > straight_threshold, 1,
                                 np.where(predictions < -straight_threshold, -1, 0))
        target_directions = np.where(targets > straight_threshold, 1,
                                   np.where(targets < -straight_threshold, -1, 0))
        return float(np.mean(pred_directions == target_directions))
    
    def _calculate_angle_accuracies_vectorized(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate angular accuracy within tolerance bands"""
        errors_deg = np.abs(np.degrees(predictions - targets))
        return {
            '5deg': float(np.mean(errors_deg <= 5.0)),
            '10deg': float(np.mean(errors_deg <= 10.0)),
            '15deg': float(np.mean(errors_deg <= 15.0))
        }
    
    def _calculate_overall_score(self, steer_metrics: Dict, throttle_metrics: Dict, brake_metrics: Dict) -> float:
        """Calculate overall control quality score (0-100)"""
        # Weighted combination of R² scores
        steer_score = max(0, steer_metrics.get('r2_score', 0)) * 40  # 40% weight
        throttle_score = max(0, throttle_metrics.get('r2_score', 0)) * 30  # 30% weight  
        brake_score = max(0, brake_metrics.get('r2_score', 0)) * 30  # 30% weight
        
        return steer_score + throttle_score + brake_score  # Already in 0-100 range


class AsyncMultiControlVisualizer:
    """Asynchronous visualizer for multi-control outputs"""
    
    def __init__(self, save_results: bool = True):
        self.save_results = save_results
        self.save_dir = Path("inference_results")
        self.save_dir.mkdir(exist_ok=True)
        
        # Load steering wheel once
        self.steering_wheel_img = self._load_steering_wheel()
        
        # Async queues
        self.vis_queue = Queue(maxsize=10)
        self.running = True
        
        # Start visualization thread
        self.vis_thread = threading.Thread(target=self._visualization_worker, daemon=True)
        self.vis_thread.start()
        
    def _load_steering_wheel(self):
        try:
            img_path = Path("steering_wheel_image.jpg")
            if img_path.exists():
                return cv2.imread(str(img_path), 0)
            else:
                # Create steering wheel visualization
                img = np.zeros((300, 300), dtype=np.uint8)
                cv2.circle(img, (150, 150), 100, 255, 3)
                cv2.line(img, (150, 50), (150, 250), 255, 2)
                cv2.line(img, (50, 150), (250, 150), 255, 2)
                return img
        except Exception as e:
            print(f"Warning: Could not load steering wheel image: {e}")
            return np.zeros((300, 300), dtype=np.uint8)
    
    def add_visualization(self, image: np.ndarray, predictions: Dict, targets: Dict):
        """Non-blocking add to visualization queue"""
        if not self.vis_queue.full():
            self.vis_queue.put((image.copy(), predictions.copy(), targets.copy()))
    
    def _visualization_worker(self):
        """Background thread for multi-control visualizations"""
        while self.running:
            try:
                if not self.vis_queue.empty():
                    image, predictions, targets = self.vis_queue.get(timeout=0.1)
                    
                    # Display input image
                    cv2.imshow("Camera Input", cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    
                    # Create steering visualizations
                    pred_steering_deg = np.degrees(predictions['steering'])
                    target_steering_deg = np.degrees(targets['steering'])
                    
                    pred_wheel = self._create_steering_visualization(pred_steering_deg)
                    target_wheel = self._create_steering_visualization(target_steering_deg)
                    
                    # Create control panel
                    control_panel = self._create_control_panel(predictions, targets)
                    
                    cv2.imshow("Predicted Steering", pred_wheel)
                    cv2.imshow("Target Steering", target_wheel)
                    cv2.imshow("Multi-Control Dashboard", control_panel)
                    
                    cv2.waitKey(1)
                else:
                    time.sleep(0.001)
            except Exception as e:
                continue
    
    def _create_steering_visualization(self, angle_degrees: float) -> np.ndarray:
        """Create steering wheel visualization"""
        if self.steering_wheel_img is None:
            return np.zeros((300, 300), dtype=np.uint8)
        rows, cols = self.steering_wheel_img.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), -angle_degrees, 1)
        return cv2.warpAffine(self.steering_wheel_img, rotation_matrix, (cols, rows))
    
    def _create_control_panel(self, predictions: Dict, targets: Dict) -> np.ndarray:
        """Create control dashboard showing all three outputs"""
        panel = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(panel, "Multi-Control Dashboard", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        y_offset = 80
        
        # Steering
        steer_pred_deg = np.degrees(predictions['steering'])
        steer_target_deg = np.degrees(targets['steering'])
        cv2.putText(panel, f"Steering:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Pred: {steer_pred_deg:6.2f}°", (10, y_offset + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(panel, f"Target: {steer_target_deg:6.2f}°", (10, y_offset + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Throttle
        y_offset += 120
        cv2.putText(panel, f"Throttle:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Pred: {predictions['throttle']:6.3f}", (10, y_offset + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(panel, f"Target: {targets['throttle']:6.3f}", (10, y_offset + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Throttle bar
        throttle_bar_height = int(predictions['throttle'] * 150)
        cv2.rectangle(panel, (300, y_offset + 50), (330, y_offset + 50 - throttle_bar_height), (0, 255, 0), -1)
        cv2.rectangle(panel, (300, y_offset - 100), (330, y_offset + 50), (100, 100, 100), 2)
        
        # Brake
        y_offset += 120
        cv2.putText(panel, f"Brake:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Pred: {predictions['brake']:6.3f}", (10, y_offset + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(panel, f"Target: {targets['brake']:6.3f}", (10, y_offset + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Brake bar
        brake_bar_height = int(predictions['brake'] * 150)
        cv2.rectangle(panel, (350, y_offset + 50), (380, y_offset + 50 - brake_bar_height), (0, 0, 255), -1)
        cv2.rectangle(panel, (350, y_offset - 100), (380, y_offset + 50), (100, 100, 100), 2)
        
        # Add labels
        cv2.putText(panel, "T", (305, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(panel, "B", (355, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return panel
    
    def save_metrics_report(self, metrics: Dict, model_name: str = "MultiControlAutonomousModel"):
        """Save comprehensive metrics report"""
        if not self.save_results:
            return
        
        def save_worker():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = self.save_dir / f"multi_control_metrics_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        
        threading.Thread(target=save_worker, daemon=True).start()
    
    def cleanup(self):
        self.running = False
        if self.vis_thread.is_alive():
            self.vis_thread.join(timeout=1.0)


def load_model_checkpoint(checkpoint_path: str, model, device):
    """Smart model loading that handles different checkpoint formats"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if it's a training checkpoint (with extra keys)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Print training info if available
            if 'val_losses' in checkpoint:
                val_losses = checkpoint['val_losses']
                print(f"✅ Multi-control model loaded with validation losses:")
                print(f"   🎯 Steering: {val_losses.get('steering', 'unknown'):.6f}")
                print(f"   ⚡ Throttle: {val_losses.get('throttle', 'unknown'):.6f}")
                print(f"   🛑 Brake: {val_losses.get('brake', 'unknown'):.6f}")
                print(f"   📊 Total: {val_losses.get('total', 'unknown'):.6f}")
            
            if 'epoch' in checkpoint:
                print(f"   📈 Trained for {checkpoint['epoch']} epochs")
                
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            print("✅ Multi-control model loaded successfully")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def main():
    """Ultra-fast multi-control evaluation"""
    
    print("🚗 Initializing MULTI-CONTROL Autonomous Driving Model Evaluator...")
    print(f"💻 Detected: {mp.cpu_count()} CPU cores, utilizing 28 workers")
    print(f"🧠 RAM: Optimized for 64GB with large batch processing")
    print("🎯 Evaluating: Steering + Throttle + Brake Control")
    
    # Load dataset with multi-output support
    dataset = get_inference_dataset("carla_001", multi_output=True)
    
    # Optimized dataloader
    dataloader = OptimizedDataLoader(
        dataset, 
        batch_size=128,  # Adjust for multi-output
        num_workers=24,
        prefetch_factor=4
    )
    
    print(f"📊 Dataset: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Load multi-control model
    model = MultiControlAutonomousModel(pretrained=False, freeze_features=False, use_speed_input=False)
    
    # Try to load the trained multi-control model
    checkpoint_path = "./models/carla_multi_control_best.pt"
    if not load_model_checkpoint(checkpoint_path, model, config.device):
        print(f"❌ Failed to load multi-control model from {checkpoint_path}")
        print("   Please ensure you have trained the multi-control model")
        return
    
    model.to(config.device)
    model.eval()
    
    # Enable optimizations
    if hasattr(torch, 'jit') and config.device == 'cuda':
        try:
            # Try to optimize with TorchScript (for multi-output model)
            sample_input = torch.randn(1, 3, 66, 200).to(config.device)
            traced_model = torch.jit.trace(model, sample_input)
            model = traced_model
            print("✅ Multi-control model optimized with TorchScript")
        except Exception as e:
            print(f"⚠ TorchScript optimization failed: {e}, using standard model")
    
    # Initialize fast components
    metrics_calculator = FastMultiControlMetrics()
    visualizer = AsyncMultiControlVisualizer(save_results=True)
    
    print(f"🏁 Starting MULTI-CONTROL evaluation...")
    print("Controls: 'q'=quit, 's'=save metrics, 'v'=toggle visualization")
    
    start_time = time.time()
    sample_count = 0
    show_visualization = True
    
    try:
        for batch_idx, batch_data in enumerate(dataloader):
            batch_start_time = time.time()
            
            # Unpack batch data (multi-output format)
            if len(batch_data) == 2:
                batch_images, batch_targets = batch_data
                # Convert single targets to multi-output format
                if not isinstance(batch_targets, dict):
                    batch_targets = {
                        'steering': batch_targets,
                        'throttle': torch.zeros_like(batch_targets),
                        'brake': torch.zeros_like(batch_targets)
                    }
            else:
                batch_images, batch_targets = batch_data[0], batch_data[1]
            
            # Move to device
            batch_images = batch_images.to(config.device, non_blocking=True)
            
            # MULTI-OUTPUT BATCH INFERENCE
            with torch.no_grad():
                batch_predictions = model(batch_images)  # Returns dict with steering, throttle, brake
            
            batch_inference_time = time.time() - batch_start_time
            
            # Add batch to metrics
            metrics_calculator.add_batch_predictions(
                batch_predictions, batch_targets, batch_inference_time
            )
            
            sample_count += len(batch_images)
            
            # Visualization for last sample in batch
            if show_visualization and batch_idx % 5 == 0:
                last_image = batch_images[-1].cpu().permute(1, 2, 0).numpy()
                last_image = (last_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                last_image = np.clip(last_image, 0, 1)
                
                # Get last predictions and targets
                last_predictions = {
                    'steering': batch_predictions['steering'][-1].item(),
                    'throttle': batch_predictions['throttle'][-1].item(),
                    'brake': batch_predictions['brake'][-1].item()
                }
                
                last_targets = {
                    'steering': batch_targets['steering'][-1].item() if hasattr(batch_targets['steering'][-1], 'item') else batch_targets['steering'][-1],
                    'throttle': batch_targets['throttle'][-1].item() if hasattr(batch_targets['throttle'][-1], 'item') else batch_targets['throttle'][-1],
                    'brake': batch_targets['brake'][-1].item() if hasattr(batch_targets['brake'][-1], 'item') else batch_targets['brake'][-1]
                }
                
                visualizer.add_visualization(last_image, last_predictions, last_targets)
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                samples_per_sec = sample_count / elapsed
                
                # Quick metrics calculation
                current_metrics = metrics_calculator.calculate_metrics_vectorized()
                
                print(f"\n📈 Batch {batch_idx}/{len(dataloader)} | Samples: {sample_count}")
                print(f"⚡ Speed: {samples_per_sec:.1f} samples/sec | {current_metrics.get('fps', 0):.1f} FPS")
                print(f"🎯 Steering - MAE: {current_metrics.get('steering_mae', 0):.3f}° | R²: {current_metrics.get('steering_r2_score', 0):.4f}")
                print(f"⚡ Throttle - MAE: {current_metrics.get('throttle_mae', 0):.3f} | R²: {current_metrics.get('throttle_r2_score', 0):.4f}")
                print(f"🛑 Brake - MAE: {current_metrics.get('brake_mae', 0):.3f} | R²: {current_metrics.get('brake_r2_score', 0):.4f}")
                print(f"🏆 Overall Score: {current_metrics.get('overall_control_score', 0):.1f}/100")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Stopping evaluation...")
                break
            elif key == ord('s'):
                print("💾 Saving multi-control metrics...")
                current_metrics = metrics_calculator.calculate_metrics_vectorized()
                visualizer.save_metrics_report(current_metrics, "MultiControlAutonomousModel")
                print("✓ Multi-control metrics saved!")
            elif key == ord('v'):
                show_visualization = not show_visualization
                print(f"👁 Visualization: {'ON' if show_visualization else 'OFF'}")
                
    except KeyboardInterrupt:
        print("\n⚠ Evaluation interrupted by user")
    
    finally:
        # Calculate final metrics
        total_time = time.time() - start_time
        print(f"\n🏁 Multi-control evaluation completed in {total_time:.2f} seconds")
        
        final_metrics = metrics_calculator.calculate_metrics_vectorized()
        
        if final_metrics:
            print("\n" + "="*80)
            print("🚗 MULTI-CONTROL AUTONOMOUS DRIVING RESULTS")
            print("="*80)
            print(f"📊 Samples Evaluated: {final_metrics['num_samples']:,}")
            print(f"⚡ Total Speed: {final_metrics['num_samples']/total_time:.1f} samples/sec")
            print(f"📦 Batches Processed: {final_metrics['total_batches']}")
            print(f"📦 Avg Batch Size: {final_metrics['avg_batch_size']:.1f}")
            print(f"⚡ Average FPS: {final_metrics['fps']:.1f}")
            
            print(f"\n🎯 STEERING CONTROL:")
            print(f"   MAE: {final_metrics.get('steering_mae', 0):.3f}°")
            print(f"   R² Score: {final_metrics.get('steering_r2_score', 0):.6f}")
            print(f"   Direction Accuracy: {final_metrics.get('direction_accuracy', 0):.1%}")
            print(f"   ±5° Accuracy: {final_metrics.get('angle_accuracy_5deg', 0):.1%}")
            print(f"   ±10° Accuracy: {final_metrics.get('angle_accuracy_10deg', 0):.1%}")
            
            print(f"\n⚡ THROTTLE CONTROL:")
            print(f"   MAE: {final_metrics.get('throttle_mae', 0):.6f}")
            print(f"   R² Score: {final_metrics.get('throttle_r2_score', 0):.6f}")
            print(f"   Range: {final_metrics.get('throttle_prediction_range', (0, 0))[0]:.3f} to {final_metrics.get('throttle_prediction_range', (0, 0))[1]:.3f}")
            
            print(f"\n🛑 BRAKE CONTROL:")
            print(f"   MAE: {final_metrics.get('brake_mae', 0):.6f}")
            print(f"   R² Score: {final_metrics.get('brake_r2_score', 0):.6f}")
            print(f"   Range: {final_metrics.get('brake_prediction_range', (0, 0))[0]:.3f} to {final_metrics.get('brake_prediction_range', (0, 0))[1]:.3f}")
            
            print(f"\n🏆 OVERALL CONTROL SCORE: {final_metrics.get('overall_control_score', 0):.1f}/100")
            print("="*80)
            
            # Save final report
            visualizer.save_metrics_report(final_metrics, "MultiControlAutonomousModel")
            print("✅ Final multi-control metrics report saved!")
        
        # Cleanup
        visualizer.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()