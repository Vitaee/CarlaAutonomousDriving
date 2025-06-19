import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
import os
import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Union
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedCarlaDataset(Dataset):
    """
    Enhanced dataset loader for ProductionCarlaModel
    
    Supports:
    - Temporal sequences (5 consecutive frames)
    - Multi-modal data (3 cameras + LiDAR)
    - Multi-task learning (steering, speed, emergency brake)
    - Rich vehicle state information
    - Weather and scenario diversity
    """
    
    def __init__(self, 
                 root_dir: str,
                 sequence_length: int = 5,
                 image_size: Tuple[int, int] = (192, 192),  # EfficientNet-B3 optimized size
                 augment: bool = True,
                 max_lidar_points: int = 1000):
        
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.augment = augment
        self.max_lidar_points = max_lidar_points
        
        # Find all sequence directories
        self.sequence_dirs = self._find_sequences()
        
        # Setup data augmentation
        self._setup_transforms()
        
        logger.info(f"Loaded {len(self.sequence_dirs)} sequences from {root_dir}")
        if len(self.sequence_dirs) > 0:
            self._print_dataset_stats()
    
    def _find_sequences(self) -> List[Path]:
        """Find all valid sequence directories"""
        sequence_dirs = []
        
        for seq_dir in self.root_dir.glob("sequence_*"):
            if seq_dir.is_dir():
                # Check if sequence has required files
                metadata_file = seq_dir / "metadata.json"
                if metadata_file.exists():
                    # Check if all camera directories exist
                    camera_dirs = [
                        seq_dir / "images_center",
                        seq_dir / "images_left", 
                        seq_dir / "images_right"
                    ]
                    lidar_dir = seq_dir / "lidar"
                    
                    if all(d.exists() for d in camera_dirs) and lidar_dir.exists():
                        sequence_dirs.append(seq_dir)
        
        return sorted(sequence_dirs)
    
    def _setup_transforms(self):
        """Setup data augmentation transforms"""
        if self.augment:
            self.transform = A.ReplayCompose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.6),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                ], p=0.4),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.MotionBlur(blur_limit=3, p=0.3),
                ], p=0.2),
                A.HorizontalFlip(p=0.3),  # Reduced probability for driving
                # Removed RandomFog due to parameter compatibility issues
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        # Sample some sequences to get statistics
        sample_size = min(100, len(self.sequence_dirs))
        sample_indices = random.sample(range(len(self.sequence_dirs)), sample_size)
        
        steering_angles = []
        speeds = []
        emergency_brakes = 0
        weather_counts = {}
        
        for idx in sample_indices:
            metadata_path = self.sequence_dirs[idx] / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            for frame_data in metadata:
                vehicle_state = frame_data['vehicle_state']
                steering_angles.append(vehicle_state['steering'])
                speeds.append(vehicle_state['speed_kmh'])
                emergency_brakes += frame_data['emergency_brake']
                
                weather = frame_data.get('weather', 'unknown')
                weather_counts[weather] = weather_counts.get(weather, 0) + 1
        
        if steering_angles:
            logger.info(f"  Dataset statistics (sample of {sample_size} sequences):")
            logger.info(f"    Steering range: [{min(steering_angles):.3f}, {max(steering_angles):.3f}]")
            logger.info(f"    Speed range: [{min(speeds):.1f}, {max(speeds):.1f}] km/h")
            logger.info(f"    Emergency brake scenarios: {emergency_brakes} ({emergency_brakes/len(steering_angles)*100:.1f}%)")
            logger.info(f"    Weather distribution: {weather_counts}")
    
    def _load_sequence_metadata(self, sequence_dir: Path) -> List[Dict]:
        """Load metadata for a sequence"""
        metadata_path = sequence_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and validate image"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_lidar(self, lidar_path: Path) -> np.ndarray:
        """Load and process LiDAR data"""
        points = np.load(lidar_path)
        
        # Ensure consistent shape
        if len(points) > self.max_lidar_points:
            # Random subsample
            indices = np.random.choice(len(points), self.max_lidar_points, replace=False)
            points = points[indices]
        elif len(points) < self.max_lidar_points:
            # Pad with zeros
            padding = np.zeros((self.max_lidar_points - len(points), 3))
            points = np.vstack([points, padding])
        
        return points.astype(np.float32)
    
    def _apply_augmentation_to_sequence(self, images_dict: Dict[str, List[np.ndarray]]) -> Tuple[Dict[str, torch.Tensor], bool]:
        """Apply consistent augmentation across all cameras and frames"""
        # Apply same transform to all frames in sequence
        transformed_images = {camera: [] for camera in ['center', 'left', 'right']}
        steering_flip = False
        
        # Get replay data from first center image to ensure consistency
        center_transformed = self.transform(image=images_dict['center'][0])
        
        # Check if flip was applied
        replay_data = center_transformed.get('replay', {})
        if replay_data:
            for transform_info in replay_data.get('transforms', []):
                if transform_info['__class_fullname__'] == 'HorizontalFlip' and transform_info['applied']:
                    steering_flip = True
                    break
        
        # Apply same transforms to all frames
        for frame_idx in range(len(images_dict['center'])):
            for camera in ['center', 'left', 'right']:
                if self.augment and replay_data:
                    # Apply recorded transforms
                    transformed = A.ReplayCompose.replay(center_transformed['replay'], image=images_dict[camera][frame_idx])
                else:
                    transformed = self.transform(image=images_dict[camera][frame_idx])
                
                transformed_images[camera].append(transformed['image'])
        
        # Stack frames into tensors
        for camera in transformed_images:
            transformed_images[camera] = torch.stack(transformed_images[camera])
        
        return transformed_images, steering_flip
    
    def __len__(self):
        return len(self.sequence_dirs)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a temporal sequence with all modalities
        
        Returns:
            Dict containing:
            - center_imgs: (T, 3, H, W)
            - left_imgs: (T, 3, H, W) 
            - right_imgs: (T, 3, H, W)
            - lidar_points: (T, N, 3)
            - steering: (T,) - steering angles for each frame
            - speed: (T,) - normalized speeds [0, 1]
            - emergency_brake: (T,) - emergency brake labels
            - vehicle_state: Dict with full vehicle state
        """
        sequence_dir = self.sequence_dirs[idx]
        metadata = self._load_sequence_metadata(sequence_dir)
        
        # Load images for all cameras and frames
        images_dict = {'center': [], 'left': [], 'right': []}
        lidar_points = []
        steering_angles = []
        speeds = []
        emergency_brakes = []
        vehicle_states = []
        
        for frame_idx, frame_data in enumerate(metadata):
            # Load images
            for camera in ['center', 'left', 'right']:
                image_filename = f"{camera}_{frame_idx:02d}.png"
                image_path = sequence_dir / f"images_{camera}" / image_filename
                image = self._load_image(image_path)
                images_dict[camera].append(image)
            
            # Load LiDAR
            lidar_filename = f"lidar_{frame_idx:02d}.npy"
            lidar_path = sequence_dir / "lidar" / lidar_filename
            lidar = self._load_lidar(lidar_path)
            lidar_points.append(lidar)
            
            # Extract labels and states
            vehicle_state = frame_data['vehicle_state']
            steering_angles.append(vehicle_state['steering'])
            speeds.append(vehicle_state['speed_kmh'])
            emergency_brakes.append(frame_data['emergency_brake'])
            vehicle_states.append(vehicle_state)
        
        # Apply augmentation consistently across sequence
        transformed_images, steering_flip = self._apply_augmentation_to_sequence(images_dict)
        
        # Convert to tensors
        lidar_tensor = torch.tensor(np.array(lidar_points), dtype=torch.float32)
        steering_tensor = torch.tensor(steering_angles, dtype=torch.float32)
        
        # Apply steering flip if horizontal flip was applied
        if steering_flip:
            steering_tensor = -steering_tensor
        
        # Normalize speed to [0, 1] range (assuming max speed of 120 km/h)
        speed_tensor = torch.tensor(speeds, dtype=torch.float32) / 120.0
        speed_tensor = torch.clamp(speed_tensor, 0.0, 1.0)
        
        emergency_brake_tensor = torch.tensor(emergency_brakes, dtype=torch.long)
        
        return {
            'center_imgs': transformed_images['center'],
            'left_imgs': transformed_images['left'],
            'right_imgs': transformed_images['right'],
            'lidar_points': lidar_tensor,
            'steering': steering_tensor,
            'speed': speed_tensor,
            'emergency_brake': emergency_brake_tensor,
            'vehicle_states': vehicle_states,
            'sequence_dir': str(sequence_dir)  # For debugging
        }

class LegacyCarlaDataset(Dataset):
    """Legacy dataset loader for backward compatibility"""
    
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
    
    def _create_balanced_sampler(self):
        """Create a weighted sampler to balance steering angle distribution"""
        steering_angles = np.array(self.data['steering_angle'].values, dtype=np.float32)
        
        # Create histogram of steering angles
        hist, bins = np.histogram(steering_angles, bins=41, range=(-0.4, 0.4))
        
        # Get bin indices for each steering angle
        bin_indices = np.digitize(steering_angles, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
        
        # Calculate weights (inverse frequency)
        weights = 1.0 / (hist[bin_indices] + 1e-6)
        weights = weights / weights.sum() * len(weights)
        
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
        
        # Get steering angle
        steering_angle = float(row['steering_angle'])
        
        # Apply transforms
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # Check for horizontal flip
        replay_data = transformed.get('replay', {})
        if replay_data:
            for transform_info in replay_data.get('transforms', []):
                if transform_info['__class_fullname__'] == 'HorizontalFlip' and transform_info['applied']:
                    steering_angle = -steering_angle
                    break
        
        return image, torch.tensor(steering_angle, dtype=torch.float32)

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
                        steering_angle_rad = np.radians(steering_angle)
                        
                        image_path = self.root_dir / filename
                        if image_path.exists():
                            self.data.append({
                                'filename': filename,
                                'steering_angle': steering_angle_rad,
                                'image_path': image_path
                            })
        
        self.transform = A.Compose([
            A.Resize(66, 200),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        print(f"Loaded {len(self.data)} real-world samples from {root_dir}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image = cv2.imread(str(item['image_path']))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {item['image_path']}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image = transformed['image']
        
        steering_angle = item['steering_angle']
        
        return image, torch.tensor(steering_angle, dtype=torch.float32)

def get_enhanced_dataset(root_dir: str, 
                        sequence_length: int = 5,
                        image_size: Tuple[int, int] = (192, 192),
                        augment: bool = True) -> EnhancedCarlaDataset:
    """Get enhanced dataset for ProductionCarlaModel"""
    return EnhancedCarlaDataset(
        root_dir=root_dir,
        sequence_length=sequence_length,
        image_size=image_size,
        augment=augment
    )

def get_enhanced_dataloader(root_dir: str,
                           batch_size: int = 2,  # Optimized for RTX 4070
                           num_workers: int = 4,  # Reduced for efficiency
                           shuffle: bool = True,
                           sequence_length: int = 5,
                           image_size: Tuple[int, int] = (192, 192),
                           augment: bool = True) -> DataLoader:
    """Get DataLoader for enhanced dataset"""
    
    dataset = get_enhanced_dataset(
        root_dir=root_dir,
        sequence_length=sequence_length,
        image_size=image_size,
        augment=augment
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )

def collate_temporal_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for temporal sequences"""
    
    # Stack all sequences
    collated = {}
    
    for key in batch[0].keys():
        if key == 'vehicle_states' or key == 'sequence_dir':
            # Keep as list for these
            collated[key] = [item[key] for item in batch]
        else:
            # Stack tensors
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated

# Legacy compatibility functions
def get_inference_dataset(dataset_type='enhanced'):
    """Get dataset for inference - updated to support enhanced datasets"""
    if dataset_type == 'enhanced':
        # Look for enhanced datasets
        enhanced_dirs = list(Path("data_real").glob("dataset_carla_enhanced_*"))
        if enhanced_dirs:
            return get_enhanced_dataset(str(enhanced_dirs[0]), augment=False)
        else:
            logger.warning("No enhanced datasets found, falling back to legacy")
            dataset_type = 'carla_001'
    
    if dataset_type == 'carla_001':
        return LegacyCarlaDataset(
            root_dir="data_weathers/dataset_carla_001_Town01",
            use_all_cameras=True
        )
    elif dataset_type == 'real_world':
        return RealWorldDataset(root_dir="driving_dataset")
    elif dataset_type == 'real_dataset':
        return LegacyCarlaDataset(
            root_dir="driving_dataset",
            use_all_cameras=True
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

def get_full_dataset_loader(dataset_type='enhanced') -> DataLoader:
    """Get full dataset loader with enhanced support"""
    if dataset_type == 'enhanced':
        enhanced_dirs = list(Path("data_real").glob("dataset_carla_enhanced_*"))
        if enhanced_dirs:
            return get_enhanced_dataloader(
                str(enhanced_dirs[0]), 
                batch_size=2,  # Small batch size for sequences
                shuffle=False,
                augment=False
            )
        else:
            logger.warning("No enhanced datasets found, falling back to legacy")
            dataset_type = 'carla_001'
    
    # Legacy datasets
    ds = get_inference_dataset(dataset_type)
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=24)

# Training utilities
def create_train_val_split(dataset: EnhancedCarlaDataset, 
                          train_ratio: float = 0.8) -> Tuple[Dataset, Dataset]:
    """Create train/validation split for enhanced dataset"""
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    # Use random split
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    return train_dataset, val_dataset

def get_training_dataloaders(root_dir: str,
                           batch_size: int = 2,
                           num_workers: int = 4,
                           train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Get training and validation dataloaders"""
    
    # Create datasets with/without augmentation
    train_dataset = get_enhanced_dataset(root_dir, augment=True)
    val_dataset = get_enhanced_dataset(root_dir, augment=False)
    
    # Create split
    train_split, _ = create_train_val_split(train_dataset, train_ratio)
    _, val_split = create_train_val_split(val_dataset, train_ratio)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_temporal_batch
    )
    
    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_temporal_batch
    )
    
    return train_loader, val_loader

# Example usage and testing
if __name__ == "__main__":
    # Test enhanced dataset loading
    enhanced_dirs = list(Path("data_real").glob("dataset_carla_enhanced_*"))
    
    if enhanced_dirs:
        print(f"Testing enhanced dataset: {enhanced_dirs[0]}")
        
        # Test dataset
        dataset = get_enhanced_dataset(str(enhanced_dirs[0]), augment=False)
        print(f"Dataset size: {len(dataset)}")
        
        # Test single item
        if len(dataset) > 0:
            item = dataset[0]
            print(f"Sample item keys: {item.keys()}")
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
            
            # Test dataloader
            dataloader = get_enhanced_dataloader(str(enhanced_dirs[0]), batch_size=2, shuffle=False)
            batch = next(iter(dataloader))
            print(f"\nBatch keys: {batch.keys()}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
    else:
        print("No enhanced datasets found. Please run the enhanced data collection script first.")

