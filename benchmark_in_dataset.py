import cv2
import numpy as np
import model
import math
import torch
import time
import scipy
from config import config
from model import NvidiaMultiOutputModel
from dataset_loader import get_inference_dataset
from torchvision import transforms
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


def angel_to_steer(degrees, cols, rows, smoothed_angle):
    # Avoid division by zero if degrees == smoothed_angle
    if abs(degrees - smoothed_angle) < 1e-6:
        # If angles are virtually the same, no change needed or direct assignment
        # Depending on desired behavior, smoothed_angle might just become degrees
        # For now, let's assume no change if they are too close to prevent instability
        pass # Or smoothed_angle = degrees
    else:
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    return mat, smoothed_angle


transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((200,66), antialias=True),
    transforms.Normalize(config.mean, config.std)
])



def calculate_and_plot_multi_output_performance(all_predictions, all_actuals, output_dir="."):
    """
    Calculate comprehensive performance metrics for multi-output model
    """
    if not all_predictions or not all_actuals:
        print("No data provided for performance calculation.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy arrays
    pred_steering = np.array([p['steering'] for p in all_predictions])
    pred_throttle = np.array([p['throttle'] for p in all_predictions])
    pred_brake = np.array([p['brake'] for p in all_predictions])
    
    actual_steering = np.array([a['steering'] for a in all_actuals])
    actual_throttle = np.array([a['throttle'] for a in all_actuals])
    actual_brake = np.array([a['brake'] for a in all_actuals])

    # Calculate metrics for each output
    outputs = {
        'steering': (pred_steering, actual_steering, 'Steering Angle'),
        'throttle': (pred_throttle, actual_throttle, 'Throttle'),
        'brake': (pred_brake, actual_brake, 'Brake')
    }
    
    print("\n" + "="*60)
    print("MULTI-OUTPUT MODEL PERFORMANCE METRICS")
    print("="*60)
    
    overall_metrics = {}
    
    for output_name, (pred, actual, display_name) in outputs.items():
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        rmse = np.sqrt(mse)
        
        # Store for overall calculation
        overall_metrics[output_name] = {'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse}
        
        print(f"\n{display_name.upper()}:")
        print(f"  Samples: {len(actual)}")
        print(f"  MSE:     {mse:.6f}")
        print(f"  RMSE:    {rmse:.6f}")
        print(f"  MAE:     {mae:.6f}")
        print(f"  R² Score: {r2:.6f}")
        print(f"  Range:   [{actual.min():.3f}, {actual.max():.3f}]")
        
        # Individual scatter plots
        plt.figure(figsize=(8, 6))
        plt.scatter(actual, pred, alpha=0.6, edgecolors='k', s=30)
        min_val = min(actual.min(), pred.min()) - 0.05
        max_val = max(actual.max(), pred.max()) + 0.05
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel(f"Actual {display_name}")
        plt.ylabel(f"Predicted {display_name}")
        plt.title(f"{display_name}: Predicted vs Actual\nR² = {r2:.4f}, RMSE = {rmse:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"multi_output_{output_name}_scatter.png"), dpi=150)
        plt.close()
        
        # Error distribution plots
        errors = pred - actual
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.75, color='coral', edgecolor='black')
        plt.xlabel(f"{display_name} Prediction Error")
        plt.ylabel("Frequency")
        plt.title(f"{display_name} Prediction Error Distribution")
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        plt.axvline(mean_err, color='r', linestyle='dashed', linewidth=2, 
                   label=f'Mean: {mean_err:.4f}')
        plt.axvline(0, color='g', linestyle='solid', linewidth=2, alpha=0.7, 
                   label='Perfect Prediction')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"multi_output_{output_name}_errors.png"), dpi=150)
        plt.close()

    # Combined time series plot
    plt.figure(figsize=(15, 10))
    sample_indices = np.arange(len(actual_steering))
    
    # Steering subplot
    plt.subplot(3, 1, 1)
    plt.plot(sample_indices, actual_steering, label='Actual', color='blue', alpha=0.8, linewidth=1)
    plt.plot(sample_indices, pred_steering, label='Predicted', color='red', alpha=0.8, linewidth=1)
    plt.ylabel('Steering Angle')
    plt.title('Multi-Output Model Performance Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Throttle subplot
    plt.subplot(3, 1, 2)
    plt.plot(sample_indices, actual_throttle, label='Actual', color='blue', alpha=0.8, linewidth=1)
    plt.plot(sample_indices, pred_throttle, label='Predicted', color='red', alpha=0.8, linewidth=1)
    plt.ylabel('Throttle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Brake subplot
    plt.subplot(3, 1, 3)
    plt.plot(sample_indices, actual_brake, label='Actual', color='blue', alpha=0.8, linewidth=1)
    plt.plot(sample_indices, pred_brake, label='Predicted', color='red', alpha=0.8, linewidth=1)
    plt.ylabel('Brake')
    plt.xlabel('Sample Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_output_time_series.png"), dpi=150)
    plt.close()

    # Overall performance summary
    print(f"\n{'='*60}")
    print("OVERALL MODEL SUMMARY:")
    print(f"{'='*60}")
    avg_r2 = np.mean([metrics['r2'] for metrics in overall_metrics.values()])
    avg_rmse = np.mean([metrics['rmse'] for metrics in overall_metrics.values()])
    print(f"Average R² Score: {avg_r2:.4f}")
    print(f"Average RMSE: {avg_rmse:.6f}")
    print(f"Total Samples: {len(all_predictions)}")
    print(f"Plots saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    # --- Data Collection for Multi-Output Performance Analysis ---
    all_predictions = []  # Will store dict with steering, throttle, brake
    all_actuals = []      # Will store dict with steering, throttle, brake
    # --- End Data Collection ---

    dataset = get_inference_dataset("carla_001", False)
    dataset_iterator = iter(dataset)

    # Load multi-output model
    model_instance = NvidiaMultiOutputModel()  # Changed model
    model_instance.load_state_dict(torch.load("./save_multioutput/model_multioutput.pt", map_location=torch.device(config.device), weights_only=True))
    model_instance.to(config.device)
    model_instance.eval()

    # Check if steering wheel image exists
    steering_wheel_path = './steering_wheel_image.jpg'
    if not os.path.exists(steering_wheel_path):
        print(f"Warning: Steering wheel image not found at {steering_wheel_path}. Steering wheel visualization will be skipped.")
        steering_wheel_1 = None
        steering_wheel_2 = None
        rows, cols = 0, 0
    else:
        steering_wheel_1 = cv2.imread(steering_wheel_path, 0)
        steering_wheel_2 = steering_wheel_1.copy()
        rows, cols = steering_wheel_1.shape

    smoothed_angle_1 = 1e-10
    smoothed_angle_2 = 1e-10

    print("Processing dataset for multi-output inference and visualization. Press 'Q' in the OpenCV window to stop early.")
    sample_count = 0
    
    try:
        while True:
            key = cv2.waitKey(20)
            if key == ord('q'):
                print("User pressed 'Q'. Stopping data processing.")
                break
            
            try:
                image, targets = next(dataset_iterator)  # targets is now a dict
                sample_count += 1
            except StopIteration:
                print("Dataset exhausted.")
                break

            # Prepare image for model input
            transformed_image_for_model = image.to(config.device)
            batch_t = torch.unsqueeze(transformed_image_for_model, 0)

            # Multi-output predictions
            with torch.no_grad():
                predictions = model_instance(batch_t)  # Returns dict with steering, throttle, brake

            # Extract predictions (convert to float)
            pred_steering = predictions['steering'].item()
            pred_throttle = predictions['throttle'].item()
            pred_brake = predictions['brake'].item()

            # Extract actual values
            actual_steering = targets['steering'].item() if torch.is_tensor(targets['steering']) else targets['steering']
            actual_throttle = targets['throttle'].item() if torch.is_tensor(targets['throttle']) else targets['throttle']
            actual_brake = targets['brake'].item() if torch.is_tensor(targets['brake']) else targets['brake']

            # Store predictions and actuals
            all_predictions.append({
                'steering': pred_steering,
                'throttle': pred_throttle,
                'brake': pred_brake
            })
            
            all_actuals.append({
                'steering': actual_steering,
                'throttle': actual_throttle,
                'brake': actual_brake
            })
            
            if sample_count % 50 == 0:
                print(f"Sample {sample_count}:")
                print(f"  Steering - Pred: {pred_steering:.4f}, Actual: {actual_steering:.4f}")
                print(f"  Throttle - Pred: {pred_throttle:.4f}, Actual: {actual_throttle:.4f}")
                print(f"  Brake    - Pred: {pred_brake:.4f}, Actual: {actual_brake:.4f}")

            # ---- Prepare image for display ----
            display_tensor = image.clone()
            mean_tensor = torch.tensor(config.mean, device=image.device).view(-1, 1, 1)
            std_tensor = torch.tensor(config.std, device=image.device).view(-1, 1, 1)
            display_tensor = display_tensor * std_tensor + mean_tensor
            display_tensor = torch.clamp(display_tensor, 0, 1)
            display_numpy_rgb = display_tensor.permute(1, 2, 0).cpu().numpy()
            frame_for_display = (display_numpy_rgb * 255).astype(np.uint8)
            
            # Add text overlay with predictions
            cv2.putText(frame_for_display, f"Steering: {pred_steering:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_for_display, f"Throttle: {pred_throttle:.3f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_for_display, f"Brake: {pred_brake:.3f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Multi-Output Predictions", cv2.cvtColor(frame_for_display, cv2.COLOR_RGB2BGR))
            # ---- End of display preparation ----

            # Steering wheel visualization (only for steering)
            if steering_wheel_1 is not None:
                pred_degrees = np.degrees(pred_steering)
                actual_degrees = np.degrees(actual_steering)
                
                mat_1, smoothed_angle_1 = angel_to_steer(pred_degrees, cols, rows, smoothed_angle_1)
                dst_1 = cv2.warpAffine(steering_wheel_1, mat_1, (cols, rows))
                cv2.imshow("Pred steering wheel", dst_1)

                mat_2, smoothed_angle_2 = angel_to_steer(actual_degrees, cols, rows, smoothed_angle_2)
                dst_2 = cv2.warpAffine(steering_wheel_2, mat_2, (cols, rows))
                cv2.imshow("Target steering wheel", dst_2)

    finally:
        cv2.destroyAllWindows()
        print(f"\nFinished processing {sample_count} samples from the dataset.")
        
        # --- Perform multi-output analysis and plotting ---
        if all_predictions and all_actuals:
            calculate_and_plot_multi_output_performance(all_predictions, all_actuals, output_dir="./visualizations")
        else:
            print("No data collected to calculate performance (or script was quit too early).")


if __name__ == "__main__":
    main()
