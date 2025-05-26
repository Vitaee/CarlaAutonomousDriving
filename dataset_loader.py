from typing import Tuple
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from config import config
import pandas as pd
import numpy as np
import os
import torch
import cv2


def add_random_shadow_bgr(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    intensity = 0.4
    x1, x2 = np.random.randint(0, img.shape[1], size=2)
    if x1 > x2:
        x1, x2 = x2, x1
    hsv[:, x1:x2, 2] = hsv[:, x1:x2, 2] * intensity
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_random_brightness_bgr(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    offset = np.random.randint(-50, 50)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + offset, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def convert_opencv_image_to_torch(image):
    
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float()
    image = (image / 127.5) - 1.0

    return image


class CarlaSimulatorDataset(Dataset):
    def __init__(self, 
                 csv_file: str = "steering_data.csv", 
                 root_dir: str = "datasets/dataset_carla_001_Town10HD_Opt", 
                 train_only_center: bool = True):
        self.dataset_folder = root_dir
        csv_path = os.path.join(self.dataset_folder, csv_file)
        self.data = pd.read_csv(csv_path)
        self.train_only_center = train_only_center

        if self.train_only_center:
            self.data = self.data[self.data['camera_position'] == 'center'].reset_index(drop=True)
            print(f"Filtered to center camera only: {len(self.data)} samples")
    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row_idx = idx // 3
        row = self.data.iloc[row_idx]

        filename = row['frame_filename']
        camera_pos = row['camera_position']
        
        if self.train_only_center:
            img_dir = os.path.join(self.dataset_folder, "images_center")
        else:
            img_dir = os.path.join(self.dataset_folder, f"images_{camera_pos}")

        img_path = os.path.join(img_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            #raise FileNotFoundError(f"Could not load image at {img_path}")
            print(f"Warning: Could not load image at {img_path}. Skipping...")
            # Return a black image as fallback or skip to next valid image
            if hasattr(self, '_last_valid_image'):
                image = self._last_valid_image.copy()
            else:
                image = np.zeros((480, 640, 3), dtype=np.uint8)  # Default black image
        else:
            self._last_valid_image = image.copy()  # Store for fallback

        # Load all control values
        steering = round(float(row['steering_angle']), 4)
        throttle = round(float(row['throttle']), 4)
        brake = round(float(row['brake']), 4)

        # Apply augmentations
        aug_type = idx % 3
        if aug_type == 1:
            image = add_random_brightness_bgr(image)
        elif aug_type == 2:
            image = cv2.flip(image, 1)
            steering = -steering  # Only flip steering

        # Process image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image = cv2.resize(image, (200, 66))
        torch_image = convert_opencv_image_to_torch(image)
        
        # Return image and control dict
        controls = {
            'steering': steering,
            'throttle': throttle,
            'brake': brake
        }
        
        return torch_image, controls

def get_inference_dataset(dataset_type='carla_001', train_only_center=True):
    if dataset_type == 'carla_001':
        return CarlaSimulatorDataset(
            root_dir="dataset_multioutput/dataset_carla_001_Town10HD_Opt",
            train_only_center=train_only_center
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")


def get_datasets(dataset_types=['carla_001']) -> Dataset:
    datasets_list = []
    for dt in dataset_types:
        datasets_list.append(get_inference_dataset(dt))
    return ConcatDataset(datasets_list)


def get_data_subsets_loaders(dataset_types=['carla_001'],
                             batch_size=config.batch_size,
                                train_only_center=True
                             ) -> Tuple[DataLoader, DataLoader]:
    
    loaded = [get_inference_dataset(dt, train_only_center) for dt in dataset_types]
    
    print(f"Total number of samples: {[len(ds) for ds in loaded]}")


    merged = ConcatDataset(loaded)
    train_set, val_set = random_split(
        merged,
        [config.train_split_size, config.test_split_size]
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=config.shuffle, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=config.shuffle, num_workers=config.num_workers
    )
    return train_loader, val_loader


def get_full_dataset_loader(dataset_type='carla_001', train_only_center=True) -> DataLoader:
    ds = get_inference_dataset(dataset_type, train_only_center)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
