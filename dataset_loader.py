import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

CAM_OFFSET = {"left": 0.10, "center": 0.0, "right": -0.10}

class CarlaDataset(Dataset):
    def __init__(self, root_dir, csv_file="steering_data.csv", use_all_cameras=True, 
                 multi_output=False, use_speed_input=False):
        self.root_dir = Path(root_dir)
        self.use_all_cameras = use_all_cameras
        self.multi_output = multi_output
        self.use_speed_input = use_speed_input
        
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
            'right': self.root_dir / 'images_right'        
        }
        
        # Data augmentation pipeline
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
        ])
        
        self.transform_val = A.Compose([
            A.Resize(66, 200),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()        
        ])
        
        # Create balanced sampler for steering angles
        self.sampler = self._create_balanced_sampler()
        
        print(f"Loaded {len(self.data)} samples from {root_dir}")
        if multi_output:
            print("  Multi-output mode: steering + throttle + brake")
        if use_speed_input:
            print("  Speed-aware mode: using current speed as input")
    
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
        
        # Get control data
        steering_angle = float(row['steering_angle'])
        throttle = float(row['throttle'])
        brake = float(row['brake'])
        speed_kmh = float(row['speed_kmh'])
        
        # Apply camera offset correction for steering
        steering_angle += CAM_OFFSET[camera_pos]
        
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
        
        # Return different formats based on model type
        if self.multi_output:
            # Multi-output model
            targets = {
                'steering': torch.tensor(steering_angle, dtype=torch.float32),
                'throttle': torch.tensor(throttle, dtype=torch.float32), 
                'brake': torch.tensor(brake, dtype=torch.float32)
            }
            
            if self.use_speed_input:
                return image, targets, torch.tensor(speed_kmh, dtype=torch.float32)
            else:
                return image, targets
        else:
            # Single-output model (steering only)
            return image, torch.tensor(steering_angle, dtype=torch.float32)


def get_inference_dataset(dataset_type='carla_001', multi_output=False, use_speed_input=False):
    if dataset_type == 'carla_001':
        return CarlaDataset(
            root_dir="data/dataset_carla_001_Town01",
            use_all_cameras=True,
            multi_output=multi_output,
            use_speed_input=use_speed_input
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

def get_full_dataset_loader(dataset_type='carla_001', multi_output=False, use_speed_input=False) -> DataLoader:
    ds = get_inference_dataset(dataset_type, multi_output, use_speed_input)
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=24)

def create_multi_output_data_loaders(batch_size=64, num_workers=24, use_all_cameras=True, use_speed_input=False):
    """Create data loaders for multi-output training"""
    
    datasets_path = Path("data")
    town_folders = [
        "dataset_carla_001_Town01",
        "dataset_carla_001_Town02", 
        "dataset_carla_001_Town03",
        "dataset_carla_001_Town04",
        "dataset_carla_001_Town05",
    ]
    
    all_datasets = []
    
    for town_folder in town_folders:
        town_path = datasets_path / town_folder
        if town_path.exists():
            print(f"Loading dataset: {town_folder}")
            dataset = CarlaDataset(
                root_dir=str(town_path),
                use_all_cameras=use_all_cameras,
                multi_output=True,
                use_speed_input=use_speed_input
            )
            all_datasets.append(dataset)
            print(f"  Loaded {len(dataset)} samples")
        else:
            print(f"Warning: {town_folder} not found")
    
    if not all_datasets:
        raise ValueError("No datasets found!")
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(all_datasets)
    print(f"Total combined samples: {len(combined_dataset)}")
    
    # Split into train/val
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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