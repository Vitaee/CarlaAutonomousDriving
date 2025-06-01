import torch
import os
from dataclasses import dataclass


@dataclass
class Config:
    # Dataset settings
    dataset_type: str = "carla_001"
    train_split_size: float = 0.8
    test_split_size: float = 0.2
    
    # Training settings
    batch_size: int = 128
    epochs_count: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001
    
    # System settings
    num_workers: int = min(12, os.cpu_count()) # type: ignore
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    is_saving_enabled: bool = True
    model_path: str = "save/model.pth"


config = Config()