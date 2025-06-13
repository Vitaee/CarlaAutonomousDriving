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
from model import NvidiaModelTransferLearning
from dataset_loader import get_inference_dataset


class CustomDataLoader:
    """High-performance dataloader optimized for my pc"""
    
    def __init__(self, dataset, batch_size=256, num_workers=28, prefetch_factor=4):
        """
        Optimized for  32-core CPU:
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


class FastAutonomousDrivingMetrics:
    """Vectorized metrics calculator for batch processing"""
    
    def __init__(self):
        self.predictions = []
        self.targets = []
        self.inference_times = []
        self.batch_sizes = []
        
    def add_batch_predictions(self, predictions: np.ndarray, targets: np.ndarray, inference_time: float):
        """Add batch predictions for faster processing"""
        self.predictions.extend(predictions.tolist())
        self.targets.extend(targets.tolist())
        batch_size = len(predictions)
        # Distribute inference time across batch
        time_per_sample = inference_time / batch_size
        self.inference_times.extend([time_per_sample] * batch_size)
        self.batch_sizes.append(batch_size)
        
    def calculate_metrics_vectorized(self) -> Dict:
        """Vectorized metric calculations for speed"""
        if not self.predictions:
            return {}
            
        # Convert to numpy arrays once
        pred_array = np.array(self.predictions, dtype=np.float32)
        target_array = np.array(self.targets, dtype=np.float32)
        
        # Vectorized basic metrics
        errors = pred_array - target_array
        abs_errors = np.abs(errors)
        
        mae = np.mean(abs_errors)
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        r2 = r2_score(target_array, pred_array)
        mape = np.mean(abs_errors / (np.abs(target_array) + 1e-8)) * 100
        
        # Vectorized autonomous driving metrics
        direction_accuracy = self._calculate_direction_accuracy_vectorized(pred_array, target_array)
        angle_accuracies = self._calculate_angle_accuracies_vectorized(pred_array, target_array)
        steering_smoothness = self._calculate_steering_smoothness_vectorized(pred_array)
        
        # Performance metrics
        avg_inference_time = np.mean(self.inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        # Error percentiles (vectorized)
        error_percentiles = np.percentile(abs_errors, [50, 75, 90, 95, 99])
        
        return {
            'mae': mae, 'mse': mse, 'rmse': rmse, 'r2_score': r2, 'mape': mape,
            'direction_accuracy': direction_accuracy,
            'angle_accuracy_5deg': angle_accuracies['5deg'],
            'angle_accuracy_10deg': angle_accuracies['10deg'],
            'angle_accuracy_15deg': angle_accuracies['15deg'],
            'steering_smoothness': steering_smoothness,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'fps': fps,
            'median_abs_error': error_percentiles[0],
            'p75_abs_error': error_percentiles[1],
            'p90_abs_error': error_percentiles[2],
            'p95_abs_error': error_percentiles[3],
            'p99_abs_error': error_percentiles[4],
            'num_samples': len(self.predictions),
            'prediction_std': np.std(pred_array),
            'target_std': np.std(target_array),
            'prediction_range': (float(np.min(pred_array)), float(np.max(pred_array))),
            'target_range': (float(np.min(target_array)), float(np.max(target_array))),
            'total_batches': len(self.batch_sizes),
            'avg_batch_size': np.mean(self.batch_sizes)
        }
    
    def _calculate_direction_accuracy_vectorized(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        straight_threshold = 0.05
        pred_directions = np.where(predictions > straight_threshold, 1,
                                 np.where(predictions < -straight_threshold, -1, 0))
        target_directions = np.where(targets > straight_threshold, 1,
                                   np.where(targets < -straight_threshold, -1, 0))
        return float(np.mean(pred_directions == target_directions))
    
    def _calculate_angle_accuracies_vectorized(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        errors_deg = np.abs(np.degrees(predictions - targets))
        return {
            '5deg': float(np.mean(errors_deg <= 5.0)),
            '10deg': float(np.mean(errors_deg <= 10.0)),
            '15deg': float(np.mean(errors_deg <= 15.0))
        }
    
    def _calculate_steering_smoothness_vectorized(self, predictions: np.ndarray) -> float:
        if len(predictions) < 2:
            return 0.0
        return float(np.var(np.diff(predictions)))


class AsyncVisualizer:
    """Asynchronous visualizer to not block main inference loop"""
    
    def __init__(self, save_results: bool = True):
        self.save_results = save_results
        self.save_dir = Path("inference_results")
        self.save_dir.mkdir(exist_ok=True)
        
        # Load steering wheel once
        self.steering_wheel_img = self._load_steering_wheel()
        
        # Async queues
        self.vis_queue = Queue(maxsize=10)  # Limit queue size to prevent memory issues
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
                # Create steering wheel
                img = np.zeros((300, 300), dtype=np.uint8)
                cv2.circle(img, (150, 150), 100, 255, 3)
                cv2.line(img, (150, 50), (150, 250), 255, 2)
                cv2.line(img, (50, 150), (250, 150), 255, 2)
                return img
        except Exception as e:
            print(f"Warning: Could not load steering wheel image: {e}")
            return np.zeros((300, 300), dtype=np.uint8)
    
    def add_visualization(self, image: np.ndarray, pred_angle: float, target_angle: float):
        """Non-blocking add to visualization queue"""
        if not self.vis_queue.full():
            self.vis_queue.put((image.copy(), pred_angle, target_angle))
    
    def _visualization_worker(self):
        """Background thread for visualizations"""
        while self.running:
            try:
                if not self.vis_queue.empty():
                    image, pred_angle, target_angle = self.vis_queue.get(timeout=0.1)
                    
                    # Display image
                    cv2.imshow("Input Image", cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    
                    # Create steering wheels
                    pred_wheel = self._create_steering_visualization(pred_angle)
                    target_wheel = self._create_steering_visualization(target_angle)
                    
                    cv2.imshow("Predicted Steering", pred_wheel)
                    cv2.imshow("Target Steering", target_wheel)
                    
                    cv2.waitKey(1)
                else:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
            except Exception as e:
                continue
    
    def _create_steering_visualization(self, angle_degrees: float) -> np.ndarray:
        if self.steering_wheel_img is None:
            return np.zeros((300, 300), dtype=np.uint8)
        rows, cols = self.steering_wheel_img.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), -angle_degrees, 1)
        return cv2.warpAffine(self.steering_wheel_img, rotation_matrix, (cols, rows))
    
    def save_metrics_report(self, metrics: Dict, model_name: str = "NvidiaModel"):
        """Async save"""
        if not self.save_results:
            return
        
        def save_worker():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = self.save_dir / f"metrics_report_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        
        threading.Thread(target=save_worker, daemon=True).start()
    
    def cleanup(self):
        self.running = False
        if self.vis_thread.is_alive():
            self.vis_thread.join(timeout=1.0)


def main():
    """Ultra-fast main function optimized for your hardware"""
    
    print("🚀 Initializing ULTRA-FAST Autonomous Driving Model Evaluator...")
    print(f"💻 Detected: {mp.cpu_count()} CPU cores, utilizing 28 workers")
    print(f"🧠 RAM: Optimized for 64GB with large batch processing")
    
    # Load dataset with optimized dataloader
    dataset = get_inference_dataset("carla_001")
    
    # Optimized dataloader for your hardware
    dataloader = CustomDataLoader(
        dataset, 
        batch_size=128,  # Large batches for your 64GB RAM
        num_workers=24,  # Utilize most of your 32 cores
        prefetch_factor=4  # High prefetch for continuous data flow
    )
    
    print(f"📊 Dataset: {len(dataset)} samples, {len(dataloader)} batches of size 256")
    
    # Load and optimize model
    model = NvidiaModelTransferLearning(pretrained=False)
    
    try:
        checkpoint = torch.load("./checkpoints/carla_steering_best.pt", map_location=torch.device(config.device))
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model loaded successfully from training checkpoint")
            if 'epoch' in checkpoint:
                print(f"  └─ Trained for {checkpoint['epoch']} epochs")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("✗ Model file not found. Please ensure './checkpoints/carla_steering_best.pt' exists")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    model.to(config.device)
    model.eval()
    
    # Enable optimizations
    if hasattr(torch, 'jit') and config.device == 'cuda':
        try:
            # Try to optimize with TorchScript
            sample_input = torch.randn(1, 3, 66, 200).to(config.device)
            model = torch.jit.trace(model, sample_input)
            print("✓ Model optimized with TorchScript")
        except:
            print("⚠ TorchScript optimization failed, using standard model")
    
    # Initialize fast components
    metrics_calculator = FastAutonomousDrivingMetrics()
    visualizer = AsyncVisualizer(save_results=True)
    
    print(f"🏁 Starting ULTRA-FAST evaluation...")
    print("Controls: 'q'=quit, 's'=save metrics, 'v'=toggle visualization")
    
    start_time = time.time()
    sample_count = 0
    show_visualization = True
    
    try:
        for batch_idx, (batch_images, batch_targets) in enumerate(dataloader):
            batch_start_time = time.time()
            
            # Move to device
            batch_images = batch_images.to(config.device, non_blocking=True)
            batch_targets = batch_targets.cpu().numpy()  # Keep on CPU for metrics
            
            # BATCH INFERENCE - Much faster than individual samples
            with torch.no_grad():
                batch_predictions = model(batch_images).cpu().numpy()
            
            batch_inference_time = time.time() - batch_start_time
            
            # Add batch to metrics (vectorized)
            metrics_calculator.add_batch_predictions(
                batch_predictions, batch_targets, batch_inference_time
            )
            
            sample_count += len(batch_predictions)
            
            # Visualization for last sample in batch (optional)
            if batch_idx % 5 == 0:  # Every 5th batch
                last_image = batch_images[-1].cpu().permute(1, 2, 0).numpy()
                last_image = (last_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                last_image = np.clip(last_image, 0, 1)
                
                pred_degrees = np.degrees(batch_predictions[-1])
                target_degrees = np.degrees(batch_targets[-1])
                
                visualizer.add_visualization(last_image, pred_degrees, target_degrees)
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                samples_per_sec = sample_count / elapsed
                
                # Quick metrics calculation
                current_metrics = metrics_calculator.calculate_metrics_vectorized()
                
                print(f"\n📈 Batch {batch_idx}/{len(dataloader)} | Samples: {sample_count}")
                print(f"⚡ Speed: {samples_per_sec:.1f} samples/sec | {current_metrics.get('fps', 0):.1f} FPS")
                print(f"📊 MAE: {np.degrees(current_metrics.get('mae', 0)):.3f}° | R²: {current_metrics.get('r2_score', 0):.4f}")
                print(f"🎯 Direction Acc: {current_metrics.get('direction_accuracy', 0):.1%} | ±5°: {current_metrics.get('angle_accuracy_5deg', 0):.1%}")
            
                            
                                
            
            # Handle keyboard input (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Stopping evaluation...")
                break
            elif key == ord('s'):
                print("💾 Saving metrics...")
                current_metrics = metrics_calculator.calculate_metrics_vectorized()
                visualizer.save_metrics_report(current_metrics, "NvidiaModelTransferLearning")
                print("✓ Metrics saved!")
            elif key == ord('v'):
                show_visualization = not show_visualization
                print(f"👁 Visualization: {'ON' if show_visualization else 'OFF'}")
                
    except KeyboardInterrupt:
        print("\n⚠ Evaluation interrupted by user")
    
    finally:
        # Calculate final metrics
        total_time = time.time() - start_time
        print(f"\n🏁 Evaluation completed in {total_time:.2f} seconds")
        
        final_metrics = metrics_calculator.calculate_metrics_vectorized()

        
        if final_metrics:
            print("\n" + "="*80)
            print("🚀 ULTRA-FAST EVALUATION RESULTS")
            print("="*80)
            print(f"📊 Samples Evaluated: {final_metrics['num_samples']:,}")
            print(f"⚡ Total Speed: {final_metrics['num_samples']/total_time:.1f} samples/sec")
            print(f"🎯 Mean Absolute Error: {np.degrees(final_metrics['mae']):.3f}°")
            print(f"📈 R² Score: {final_metrics['r2_score']:.6f}")
            print(f"🎯 Direction Accuracy: {final_metrics['direction_accuracy']:.1%}")
            print(f"✅ ±5° Accuracy: {final_metrics['angle_accuracy_5deg']:.1%}")
            print(f"✅ ±10° Accuracy: {final_metrics['angle_accuracy_10deg']:.1%}")
            print(f"⚡ Average FPS: {final_metrics['fps']:.1f}")
            print(f"📦 Batches Processed: {final_metrics['total_batches']}")
            print(f"📦 Avg Batch Size: {final_metrics['avg_batch_size']:.1f}")
            print("="*80)
            
            # Save final report
            visualizer.save_metrics_report(final_metrics, "NvidiaModelTransferLearning")
            print("✓ Final ultra-fast metrics report saved!")
        
        # Cleanup
        visualizer.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()