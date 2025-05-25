import torch
import os
from pydantic import BaseSettings


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class Config(BaseSettings):
    dataset_type: str = "carla_001"
    batch_size: int = 64
    num_workers: int = max(os.cpu_count(), 1)
    shuffle: bool = True
    train_split_size: float = 0.75
    test_split_size: float = 0.25
    resize: tuple = (66, 200)
    epochs_count: int = 45
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    momentum: float = 0.9
    save_model: bool = True
    root_path: str = "./"
    model_path: str = "model.pt"
    optimizer_path: str = "optimizer.pt"
    loss_path: str = "loss.pt"
    device: str = get_device()
    mean: list = [0.485, 0.456, 0.406]
    std: list = [0.229, 0.224, 0.225]
    epsilon: float = 0.001
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0005

    cross_validation_folds: int = 4
    
    scheduler_type: str = "multistep"
    scheduler_multistep_milestones: list = [8, 12, 16, 20, 24]
    scheduler_step_size: int = 35
    scheduler_gamma: float = 0.5
    
    is_saving_enabled: bool = True
    is_loss_logging_enabled: bool = True
    is_image_logging_enabled: bool = False
    is_learning_rate_logging_enabled: bool = True
    is_grad_avg_logging_enabled: bool = False


config = Config()
