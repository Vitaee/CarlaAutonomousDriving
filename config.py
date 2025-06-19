import torch
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Dataset settings
    dataset_type: str = "enhanced"  # Use enhanced dataset
    train_split_size: float = 0.8
    test_split_size: float = 0.2
    
    # Enhanced dataset settings
    sequence_length: int = 5  # Temporal sequence length
    image_size: tuple = field(default_factory=lambda: (192, 192))  # REDUCED from (224, 224) for memory efficiency
    max_lidar_points: int = 1000  # REDUCED from 2000 for memory efficiency
    
    # Original settings (keeping for compatibility)
    resize: tuple = field(default_factory=lambda: (192, 192))  # Updated to match image_size
    
    # Training settings for ProductionCarlaModel (Optimized for RTX 4070 8GB)
    batch_size: int = 64  # REDUCED: Was 6, causing VRAM overflow 
    gradient_accumulation_steps: int = 3  # Effective batch size = 2 * 3 = 6
    epochs_count: int = 40  # REDUCED: Focus on quality over quantity
    learning_rate: float = 3e-4  # INCREASED: Faster convergence with smaller batches
    weight_decay: float = 5e-5  # REDUCED: Less regularization for faster training
    
    # Multi-task loss weights
    steering_loss_weight: float = 2.0  # Primary task
    speed_loss_weight: float = 1.0     # Secondary task  
    brake_loss_weight: float = 1.5     # Safety critical
    uncertainty_loss_weight: float = 0.5  # Uncertainty estimation
    anomaly_loss_weight: float = 0.3   # Anomaly detection
    
    # Advanced training settings
    gradient_clip_value: float = 0.5  # REDUCED: Less aggressive clipping for faster training
    warmup_epochs: int = 2  # REDUCED: Faster warmup
    
    # Early stopping
    early_stopping_patience: int = 8  # REDUCED: Stop earlier if not improving
    early_stopping_min_delta: float = 0.001  # INCREASED: Less sensitive
    
    # System settings (Optimized for i9 32-core + 64GB RAM)
    num_workers: int = 32  # INCREASED: Better CPU utilization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data paths
    enhanced_data_dir: str = "data_real"  # Enhanced dataset directory
    town_datasets: list = field(default_factory=lambda: ["dataset_carla_enhanced_Town02", "dataset_carla_enhanced_Town03"])
    
    # Paths
    save_dir: str = "checkpoints_real"
    log_dir: str = "logs"
    is_saving_enabled: bool = True
    model_path: str = "save/enhanced_model.pth"
    
    # Safety thresholds for validation
    uncertainty_threshold: float = 0.5
    anomaly_threshold: float = 0.3
    
    # Scheduler settings
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6


config = Config()