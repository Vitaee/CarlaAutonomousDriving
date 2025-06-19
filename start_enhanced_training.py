#!/usr/bin/env python3
"""
Enhanced training script for ProductionCarlaModel
Optimized for RTX 4070 8GB with EfficientNet-B3
"""

import subprocess
import sys
from pathlib import Path
from config import config

def main():
    """Start enhanced training with optimized settings"""
    
    print("🚗 Starting Enhanced CARLA Model Training")
    print("=" * 50)
    print(f"Model: ProductionCarlaModel (EfficientNet-B3)")
    print(f"Datasets: {config.town_datasets}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Image size: {config.image_size}")
    print(f"LiDAR points: {config.max_lidar_points}")
    print(f"Epochs: {config.epochs_count}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Workers: {config.num_workers}")
    print("=" * 50)
    
    # Verify datasets exist
    data_dir = Path(config.enhanced_data_dir)
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        return
    
    missing_datasets = []
    for dataset in config.town_datasets:
        dataset_path = data_dir / dataset
        if not dataset_path.exists():
            missing_datasets.append(dataset)
    
    if missing_datasets:
        print(f"❌ Error: Missing datasets: {missing_datasets}")
        return
    
    print("✅ All datasets found")
    
    # Start training with optimized settings (enforce config values)
    cmd = [
        sys.executable, "train.py",
        "--batch_size", str(config.batch_size),  # Force config batch size
        "--epochs", str(config.epochs_count),
        "--lr", str(config.learning_rate),
        "--num_workers", str(config.num_workers),
        "--run_name", "enhanced_carla_optimized"
    ]
    
    print(f"\n🚀 Launching training with command:")
    print(" ".join(cmd))
    print("\n" + "=" * 50)
    
    # Execute training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return
    
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main() 