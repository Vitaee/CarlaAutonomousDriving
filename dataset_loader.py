import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

CAM_OFFSET = {"left": 0.15, "center": 0.0, "right": -0.15}

class RealWorldDataset(Dataset):
    """Dataset for real-world driving images with data.txt steering angles"""
    
    def __init__(self, root_dir="driving_dataset"):
        self.root_dir = Path(root_dir)
        
        # Load data.txt file
        data_file = self.root_dir / "data.txt"
        self.data = []
        
        if data_file.exists():
            with open(data_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        steering_angle = float(parts[1])
                        # Convert from degrees to radians (model expects radians)
                        steering_angle_rad = np.radians(steering_angle)
                        
                        # Check if image file exists
                        image_path = self.root_dir / filename
                        if image_path.exists():
                            self.data.append({
                                'filename': filename,
                                'steering_angle': steering_angle_rad,
                                'image_path': image_path
                            })
        
        # Data augmentation pipeline for inference (minimal processing)
        self.transform = A.Compose([
            A.Resize(66, 200),  # Nvidia model input size
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        print(f"Loaded {len(self.data)} real-world samples from {root_dir}")
        if len(self.data) > 0:
            steering_angles = [item['steering_angle'] for item in self.data]
            print(f"  Steering angle range: [{np.degrees(min(steering_angles)):.1f}°, {np.degrees(max(steering_angles)):.1f}°]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = cv2.imread(str(item['image_path']))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {item['image_path']}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image = transformed['image']
        
        steering_angle = item['steering_angle']
        
        return image, torch.tensor(steering_angle, dtype=torch.float32)

class CarlaDataset(Dataset):
    def __init__(self, root_dir, csv_file="steering_data.csv", use_all_cameras=True):
        self.root_dir = Path(root_dir)
        self.use_all_cameras = use_all_cameras
        
        # Load CSV data
        csv_path = self.root_dir / csv_file
        self.data = pd.read_csv(csv_path)
        
        # Filter data if needed
        if not use_all_cameras:
            self.data = self.data[self.data['camera_position'] == 'center'].reset_index(drop=True)
        
        # Image directories
        self.image_dirs = {
            'center': self.root_dir / 'images_center',
            'left': self.root_dir / 'images_left', 
            'right': self.root_dir / 'images_right'        }
        
        # Data augmentation pipeline - using ReplayCompose to track transformations
        self.transform = A.ReplayCompose([
            A.Resize(66, 200),  # Nvidia model input size
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]) # type: ignore
        
        self.transform_val = A.Compose([
            A.Resize(66, 200),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()        ])
          # Create balanced sampler for steering angles
        self.sampler = self._create_balanced_sampler()
        
        print(f"Loaded {len(self.data)} samples from {root_dir}")
    
    def _create_balanced_sampler(self):
        """Create a weighted sampler to balance steering angle distribution"""
        steering_angles = np.array(self.data['steering_angle'].values, dtype=np.float32)
        
        # Create histogram of steering angles
        hist, bins = np.histogram(steering_angles, bins=41, range=(-0.4, 0.4))
        
        # Get bin indices for each steering angle
        bin_indices = np.digitize(steering_angles, bins) - 1
        
        # Clip indices to valid range [0, len(hist)-1] to handle edge cases
        bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
        
        # Calculate weights (inverse frequency)
        weights = 1.0 / (hist[bin_indices] + 1e-6)
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        # Create WeightedRandomSampler
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, 
            num_samples=len(weights), 
            replacement=True
        )
        
        # Print distribution info
        print(f"  Steering angle distribution:")
        print(f"    Range: [{steering_angles.min():.3f}, {steering_angles.max():.3f}]")
        print(f"    Mean: {steering_angles.mean():.3f}, Std: {steering_angles.std():.3f}")
        print(f"    Histogram (bins={len(hist)}): min={hist.min()}, max={hist.max()}")
        print(f"    Bin indices range: [{bin_indices.min()}, {bin_indices.max()}]")
        
        return sampler
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get image path
        camera_pos = row['camera_position']
        filename = row['frame_filename']
        image_path = self.image_dirs[camera_pos] / filename
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get steering angle with camera-specific correction
        steering_angle = float(row['steering_angle'])
        # Apply camera offset correction
        #steering_angle += CAM_OFFSET[camera_pos]
        
        # Apply transforms with replay tracking
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # Check if horizontal flip was applied and adjust steering angle accordingly
        replay_data = transformed.get('replay', {})
        if replay_data:
            for transform_info in replay_data.get('transforms', []):
                if transform_info['__class_fullname__'] == 'HorizontalFlip' and transform_info['applied']:
                    steering_angle = -steering_angle
                    break
        
        return image, torch.tensor(steering_angle, dtype=torch.float32)
    


def get_inference_dataset(dataset_type='carla_001'):
    if dataset_type == 'carla_001':
        return CarlaDataset(
            root_dir="data_weathers/dataset_carla_001_Town01",
            use_all_cameras=True
        )
    elif dataset_type == 'real_world':
        return RealWorldDataset(root_dir="driving_dataset")
    elif dataset_type == 'real_dataset':
        return CarlaDataset(
            root_dir="driving_dataset",
            use_all_cameras=True
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Valid options: 'carla_001', 'real_world', 'real_dataset'")

def get_full_dataset_loader(dataset_type='carla_001') -> DataLoader:
    ds = get_inference_dataset(dataset_type)
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=24)

