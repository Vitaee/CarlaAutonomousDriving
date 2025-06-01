import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
            'right': self.root_dir / 'images_right'
        }
        
        # Data augmentation pipeline
        self.transform = A.Compose([
            A.Resize(66, 200),  # Nvidia model input size
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], p=0.3),
            A.HorizontalFlip(p=0.5),  # We'll handle steering angle adjustment manually
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]) # type: ignore
        
        self.transform_val = A.Compose([
            A.Resize(66, 200),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        print(f"Loaded {len(self.data)} samples from {root_dir}")
    
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
        
        # Apply transforms
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # Handle horizontal flip for steering angle
        # Note: Albumentations doesn't provide flip info, so we'll do it manually
        if np.random.random() < 0.4:  # 40% chance of flip
            image = torch.flip(image, dims=[2])  # Flip horizontally
            steering_angle = -steering_angle
        
        return image, torch.tensor(steering_angle, dtype=torch.float32)