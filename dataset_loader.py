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
                 multi_output=False, use_speed_input=False, enable_control_augmentation=True):
        self.root_dir = Path(root_dir)
        self.use_all_cameras = use_all_cameras
        self.multi_output = multi_output
        self.use_speed_input = use_speed_input
        self.enable_control_augmentation = enable_control_augmentation
        
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
        
        # Control augmentation parameters
        if enable_control_augmentation:
            self._setup_control_augmentation()
        
        print(f"Loaded {len(self.data)} samples from {root_dir}")
        if multi_output:
            print("  Multi-output mode: steering + throttle + brake")
        if use_speed_input:
            print("  Speed-aware mode: using current speed as input")
        if enable_control_augmentation:
            print("  🎯 Control augmentation: ENABLED for better throttle/brake diversity")
    
    def _setup_control_augmentation(self):
        """Setup control augmentation parameters for better throttle/brake training"""
        
        # Analyze current data distribution
        throttle_mean = self.data['throttle'].mean()
        throttle_std = self.data['throttle'].std()
        brake_mean = self.data['brake'].mean()
        brake_std = self.data['brake'].std()
        
        print(f"  📊 Original Throttle: μ={throttle_mean:.3f}, σ={throttle_std:.3f}")
        print(f"  📊 Original Brake: μ={brake_mean:.3f}, σ={brake_std:.3f}")
        
        # Augmentation settings based on analysis
        self.throttle_aug_params = {
            'noise_std': max(0.05, throttle_std * 0.3),  # Add realistic noise
            'shift_range': 0.15,  # Shift values for diversity
            'speed_correlation': True,  # Correlate with speed
            'scenario_variation': 0.2  # Random scenario variations
        }
        
        self.brake_aug_params = {
            'noise_std': 0.03,  # Small noise for brake
            'emergency_prob': 0.02,  # 2% emergency braking
            'gradual_prob': 0.05,  # 5% gradual braking  
            'traffic_prob': 0.03,  # 3% traffic braking
            'max_brake': 0.8  # Realistic max brake
        }
        
        print(f"  ⚙️ Throttle augmentation: noise={self.throttle_aug_params['noise_std']:.3f}")
        print(f"  ⚙️ Brake augmentation: emergency={self.brake_aug_params['emergency_prob']:.1%}")
    
    def _augment_throttle_control(self, original_throttle, speed_kmh, is_training=True):
        """Augment throttle data for better training diversity"""
        if not self.enable_control_augmentation or not is_training:
            return original_throttle
        
        augmented = original_throttle
        
        # 1. Add realistic noise based on speed
        noise_factor = 1.0 if speed_kmh > 30 else 1.5  # More variation at low speeds
        noise = np.random.normal(0, self.throttle_aug_params['noise_std'] * noise_factor)
        augmented += noise
        
        # 2. Speed-correlated adjustments
        if self.throttle_aug_params['speed_correlation']:
            if speed_kmh < 15:  # Low speed - more throttle variation
                speed_adj = np.random.uniform(-0.2, 0.3)
                augmented += speed_adj
            elif speed_kmh > 50:  # High speed - reduce throttle
                speed_adj = np.random.uniform(-0.3, -0.1)
                augmented += speed_adj
        
        # 3. Scenario variations (simulate different driving styles)
        if np.random.random() < self.throttle_aug_params['scenario_variation']:
            scenario = np.random.choice(['aggressive', 'conservative', 'eco'])
            
            if scenario == 'aggressive':
                augmented = min(1.0, augmented + 0.15)
            elif scenario == 'conservative':
                augmented = max(0.1, augmented - 0.1)
            elif scenario == 'eco':
                augmented = max(0.0, augmented - 0.2) if speed_kmh > 25 else augmented
        
        # 4. Random value shifts for diversity
        if np.random.random() < 0.15:  # 15% chance
            shift = np.random.uniform(-self.throttle_aug_params['shift_range'], 
                                    self.throttle_aug_params['shift_range'])
            augmented += shift
        
        # Clamp to valid range
        return np.clip(augmented, 0.0, 1.0)
    
    def _augment_brake_control(self, original_brake, speed_kmh, throttle_value, is_training=True):
        """Augment brake data for better training diversity"""
        if not self.enable_control_augmentation or not is_training:
            return original_brake
        
        augmented = original_brake
        
        # 1. Emergency braking scenarios
        if np.random.random() < self.brake_aug_params['emergency_prob']:
            if speed_kmh > 35:  # Only at higher speeds
                emergency_brake = np.random.uniform(0.5, self.brake_aug_params['max_brake'])
                return emergency_brake
        
        # 2. Gradual braking scenarios
        if np.random.random() < self.brake_aug_params['gradual_prob']:
            if speed_kmh > 25 and throttle_value < 0.5:  # When not accelerating
                gradual_brake = np.random.uniform(0.1, 0.4)
                augmented = max(augmented, gradual_brake)
        
        # 3. Traffic braking scenarios
        if np.random.random() < self.brake_aug_params['traffic_prob']:
            if 20 < speed_kmh < 40:  # City/traffic speeds
                traffic_brake = np.random.uniform(0.15, 0.5)
                augmented = max(augmented, traffic_brake)
        
        # 4. Speed-based braking
        if speed_kmh > 60 and np.random.random() < 0.01:  # Rare high-speed braking
            high_speed_brake = np.random.uniform(0.2, 0.6)
            augmented = max(augmented, high_speed_brake)
        
        # 5. Anti-correlation with throttle (realistic driving)
        if throttle_value > 0.7 and np.random.random() < 0.05:
            # Occasional brake while accelerating (foot confusion simulation)
            augmented = max(augmented, np.random.uniform(0.1, 0.25))
        
        # 6. Add small noise for existing brake values
        if augmented > 0:
            noise = np.random.normal(0, self.brake_aug_params['noise_std'])
            augmented += noise
        
        # Clamp to valid range
        return np.clip(augmented, 0.0, 1.0)
    
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
        
        # Determine if this is training (based on transform choice)
        is_training = hasattr(self, '_is_training') and self._is_training
        
        # Apply control augmentation if enabled and in training mode
        if self.enable_control_augmentation:
            throttle = self._augment_throttle_control(throttle, speed_kmh, is_training)
            brake = self._augment_brake_control(brake, speed_kmh, throttle, is_training)
        
        # Apply transforms with replay tracking
        if is_training:
            transformed = self.transform(image=image)
        else:
            transformed = self.transform_val(image=image)
        
        image = transformed['image']
        
        # Check if horizontal flip was applied and adjust steering angle accordingly
        if hasattr(transformed, 'replay') and transformed.get('replay', {}):
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

def create_multi_output_data_loaders(batch_size=32, num_workers=8, val_split=0.2, 
                                   use_all_cameras=True, use_speed_input=False, 
                                   enable_control_augmentation=True):
    """Create data loaders with enhanced control augmentation"""
    
    # Collect all town datasets
    data_dirs = []
    dataset_pattern = Path("data/dataset_carla_001_*")
    
    for dataset_dir in sorted(Path(".").glob(str(dataset_pattern))):
        if dataset_dir.is_dir():
            csv_file = dataset_dir / "steering_data.csv"
            if csv_file.exists():
                data_dirs.append(dataset_dir)
                print(f"Found dataset: {dataset_dir}")
    
    if not data_dirs:
        raise FileNotFoundError("No CARLA datasets found! Run data collection first.")
    
    # Combine datasets with augmentation
    all_datasets = []
    for data_dir in data_dirs:
        dataset = CarlaDataset(
            data_dir, 
            use_all_cameras=use_all_cameras,
            multi_output=True,
            use_speed_input=use_speed_input,
            enable_control_augmentation=enable_control_augmentation
        )
        all_datasets.append(dataset)
    
    # Combine all datasets
    combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
    
    # Split into train/validation
    total_samples = len(combined_dataset)
    val_samples = int(total_samples * val_split)
    train_samples = total_samples - val_samples
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_samples, val_samples],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Set training flags for augmentation
    for dataset in all_datasets:
        dataset._is_training = True  # For training augmentation
    
    # Create validation datasets without augmentation
    val_datasets = []
    for data_dir in data_dirs:
        val_ds = CarlaDataset(
            data_dir,
            use_all_cameras=use_all_cameras,
            multi_output=True,
            use_speed_input=use_speed_input,
            enable_control_augmentation=False  # No augmentation for validation
        )
        val_ds._is_training = False
        val_datasets.append(val_ds)
    
    val_combined = torch.utils.data.ConcatDataset(val_datasets)
    _, val_dataset = torch.utils.data.random_split(
        val_combined, [train_samples, val_samples],
        generator=torch.Generator().manual_seed(42)
    )
    
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
    
    print(f"\n📊  Data Loaders Created:")
    print(f"  🚂 Training samples: {len(train_dataset):,}")
    print(f"  🔍 Validation samples: {len(val_dataset):,}")
    print(f"  📦 Batch size: {batch_size}")
    print(f"  🎯 Control augmentation: {'ENABLED' if enable_control_augmentation else 'DISABLED'}")
    
    return train_loader, val_loader