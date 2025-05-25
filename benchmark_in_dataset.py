import cv2
import numpy as np
import model
import math
import torch
import time
import scipy
from config import config
from model import NvidiaModel
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


def calculate_and_plot_performance(predictions, actuals, output_dir="."):
    if not predictions or not actuals:
        print("No data provided for performance calculation.")
        return

    predictions_np = np.array(predictions)
    actuals_np = np.array(actuals)

    mse = mean_squared_error(actuals_np, predictions_np)
    mae = mean_absolute_error(actuals_np, predictions_np)
    r2 = r2_score(actuals_np, predictions_np)
    
    print("\n--- Model Performance Metrics ---")
    print(f"Number of samples: {len(actuals_np)}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Scatter Plot: Predicted vs. Actual
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals_np, predictions_np, alpha=0.6, edgecolors='k', s=50)
    min_val = min(actuals_np.min(), predictions_np.min()) - 5 # Add some padding
    max_val = max(actuals_np.max(), predictions_np.max()) + 5 # Add some padding
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
    plt.xlabel("Actual Steering Angles (degrees)")
    plt.ylabel("Predicted Steering Angles (degrees)")
    plt.title("Predicted vs. Actual Steering Angles")
    plt.legend()
    plt.grid(True)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_predicted_vs_actual_scatter.png"))
    plt.close()
    print(f"Saved scatter plot to {os.path.join(output_dir, 'inference_predicted_vs_actual_scatter.png')}")

    # 2. Histogram of Prediction Errors
    errors = predictions_np - actuals_np
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75, color='coral', edgecolor='black')
    plt.xlabel("Prediction Error (Predicted - Actual degrees)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prediction Errors")
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    plt.axvline(mean_err, color='r', linestyle='dashed', linewidth=1.5, label=f'Mean Error: {mean_err:.2f}')
    plt.text(0.05, 0.95, f'Std Dev: {std_err:.2f}', transform=plt.gca().transAxes, ha='left', va='top')
    plt.legend()
    plt.grid(axis='y', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_prediction_errors_histogram.png"))
    plt.close()
    print(f"Saved error histogram to {os.path.join(output_dir, 'inference_prediction_errors_histogram.png')}")

    # 3. Line Plot: Predicted and Actual over Samples
    plt.figure(figsize=(15, 7))
    sample_indices = np.arange(len(actuals_np))
    plt.plot(sample_indices, actuals_np, label='Actual Angles', color='dodgerblue', linestyle='-', linewidth=1.5, alpha=0.8)
    plt.plot(sample_indices, predictions_np, label='Predicted Angles', color='orangered', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.xlabel("Dataset Sample Index")
    plt.ylabel("Steering Angle (degrees)")
    plt.title("Actual vs. Predicted Steering Angles (Dataset Sequence)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_actual_vs_predicted_line.png"))
    plt.close()
    print(f"Saved line plot to {os.path.join(output_dir, 'inference_actual_vs_predicted_line.png')}")


def main():
    # --- Data Collection for Performance Analysis ---
    all_predictions_deg = []
    all_actuals_deg = []
    # --- End Data Collection ---

    dataset = get_inference_dataset("carla_001", False)
    dataset_iterator = iter(dataset)

    model_instance = NvidiaModel()
    model_instance.load_state_dict(torch.load("./save/model.pt", map_location=torch.device(config.device), weights_only=True))
    model_instance.to(config.device)
    model_instance.eval()

    # Check if steering wheel image exists
    steering_wheel_path = './steering_wheel_image.jpg'
    if not os.path.exists(steering_wheel_path):
        print(f"Warning: Steering wheel image not found at {steering_wheel_path}. Steering wheel visualization will be skipped.")
        steering_wheel_1 = None
        steering_wheel_2 = None
        rows, cols = 0, 0 # Dummy values
    else:
        steering_wheel_1 = cv2.imread(steering_wheel_path, 0)
        steering_wheel_2 = steering_wheel_1.copy()
        rows, cols = steering_wheel_1.shape

    smoothed_angle_1 = 1e-10
    smoothed_angle_2 = 1e-10

    print("Processing dataset for inference and visualization. Press 'Q' in the OpenCV window to stop early.")
    sample_count = 0
    try:
        while True: # Loop indefinitely until 'q' or dataset ends
            key = cv2.waitKey(20) # Adjusted waitKey position for early exit
            if key == ord('q'):
                print("User pressed 'Q'. Stopping data processing.")
                break
            try:
                image, target = next(dataset_iterator) # image is a (C,H,W) tensor, target is float
                sample_count += 1
            except StopIteration:
                print("Dataset exhausted.")
                break

            # Prepare image for model input
            transformed_image_for_model = image.to(config.device)
            batch_t = torch.unsqueeze(transformed_image_for_model, 0)

            # Predictions
            with torch.no_grad():
                y_predict_rad = model_instance(batch_t) # model output is in radians

            # Converting prediction to degrees
            pred_degrees = np.degrees(y_predict_rad.item())
            target_degrees = np.degrees(target) # target from dataset_loader is already in radians

            all_predictions_deg.append(pred_degrees)
            all_actuals_deg.append(target_degrees)
            
            if sample_count % 50 == 0: # Print progress every 50 samples
                print(f"Processed {sample_count} samples. Last pred: {pred_degrees:.2f}, actual: {target_degrees:.2f}")
            # else:
            #     print(f"Predicted Steering angle: {pred_degrees}")
            #     print(f"Steering angle: {pred_degrees} (pred)\\t {target_degrees} (actual)")

            # ---- Prepare image for display ----
            display_tensor = image.clone() 
            mean_tensor = torch.tensor(config.mean, device=image.device).view(-1, 1, 1) 
            std_tensor = torch.tensor(config.std, device=image.device).view(-1, 1, 1)
            display_tensor = display_tensor * std_tensor + mean_tensor 
            display_tensor = torch.clamp(display_tensor, 0, 1) 
            display_numpy_rgb = display_tensor.permute(1, 2, 0).cpu().numpy()
            frame_for_display = (display_numpy_rgb * 255).astype(np.uint8)
            cv2.imshow("frame", cv2.cvtColor(frame_for_display, cv2.COLOR_RGB2BGR))
            # ---- End of display preparation ----

            if steering_wheel_1 is not None:
                mat_1, smoothed_angle_1 = angel_to_steer(pred_degrees, cols, rows, smoothed_angle_1)
                dst_1 = cv2.warpAffine(steering_wheel_1, mat_1, (cols, rows))
                cv2.imshow("Pred steering wheel", dst_1)

                mat_2, smoothed_angle_2 = angel_to_steer(target_degrees, cols, rows, smoothed_angle_2)
                dst_2 = cv2.warpAffine(steering_wheel_2, mat_2, (cols, rows))
                cv2.imshow("Target steering wheel", dst_2)
    
    finally:
        cv2.destroyAllWindows()
        print(f"\nFinished processing {sample_count} samples from the dataset.")
        # --- Perform analysis and plotting ---
        if all_predictions_deg and all_actuals_deg:
            calculate_and_plot_performance(all_predictions_deg, all_actuals_deg, output_dir="./visualizations")
        else:
            print("No data collected to calculate performance (or script was quit too early).")
        # --- End analysis ---


if __name__ == "__main__":
    main()
