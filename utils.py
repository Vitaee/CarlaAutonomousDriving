from torchvision.utils import make_grid
import torch.nn.functional as F
import torch, cv2
import dataset_loader
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from config import config
from model import activation
from PIL import Image


def save_model(model, log_dir="./save_multioutput"):
    if not config.is_saving_enabled:
        return
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    checkpoint_path = os.path.join(log_dir, config.model_path)
    if config.device == 'cuda':
        model.to('cpu')
    torch.save(model.state_dict(), checkpoint_path)

    if config.device == 'cuda':
        model.to('cuda')


def train(desc_message, model, train_subset_loader, loss_functions, optimizer):
    model.train()
    batch_losses = {
        'steering': np.array([]),
        'throttle': np.array([]),
        'brake': np.array([]),
        'total': np.array([])
    }

    for data, targets in tqdm(train_subset_loader, desc=desc_message, ascii=' ='):
        data = data.to(config.device)
        
        # Move targets to device
        targets_device = {}
        for key, value in targets.items():
            targets_device[key] = value.to(config.device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(data)
        
        # Calculate individual losses
        steering_loss = loss_functions['steering'](
            predictions['steering'].float(), 
            targets_device['steering'].float()
        )
        throttle_loss = loss_functions['throttle'](
            predictions['throttle'].float(), 
            targets_device['throttle'].float()
        )
        brake_loss = loss_functions['brake'](
            predictions['brake'].float(), 
            targets_device['brake'].float()
        )
        
        # Weighted total loss
        total_loss = (
            config.steering_weight * steering_loss + 
            config.throttle_weight * throttle_loss + 
            config.brake_weight * brake_loss
        )

        total_loss.backward()
        optimizer.step()

        # Log losses
        batch_losses['steering'] = np.append(batch_losses['steering'], steering_loss.item())
        batch_losses['throttle'] = np.append(batch_losses['throttle'], throttle_loss.item())
        batch_losses['brake'] = np.append(batch_losses['brake'], brake_loss.item())
        batch_losses['total'] = np.append(batch_losses['total'], total_loss.item())

    return {key: losses.mean() for key, losses in batch_losses.items()}


def validation(desc_message, model, val_subset_loader, loss_functions):
    """
    Validation function for multi-output model
    """
    model.eval()
    batch_losses = {
        'steering': np.array([]),
        'throttle': np.array([]),
        'brake': np.array([]),
        'total': np.array([])
    }

    with torch.no_grad():
        for data_val, targets_val in tqdm(val_subset_loader, desc=desc_message, ascii=' ='):
            # Send data to device
            data_val = data_val.to(config.device)
            
            # Move targets to device
            targets_val_device = {}
            for key, value in targets_val.items():
                targets_val_device[key] = value.to(config.device)

            # Forward pass
            predictions_val = model(data_val)
            
            # Calculate individual losses
            steering_loss = loss_functions['steering'](
                predictions_val['steering'].float(), 
                targets_val_device['steering'].float()
            )
            throttle_loss = loss_functions['throttle'](
                predictions_val['throttle'].float(), 
                targets_val_device['throttle'].float()
            )
            brake_loss = loss_functions['brake'](
                predictions_val['brake'].float(), 
                targets_val_device['brake'].float()
            )
            
            # Weighted total loss (same as training)
            total_loss = (
                config.steering_weight * steering_loss + 
                config.throttle_weight * throttle_loss + 
                config.brake_weight * brake_loss
            )

            # Log losses
            batch_losses['steering'] = np.append(batch_losses['steering'], steering_loss.item())
            batch_losses['throttle'] = np.append(batch_losses['throttle'], throttle_loss.item())
            batch_losses['brake'] = np.append(batch_losses['brake'], brake_loss.item())
            batch_losses['total'] = np.append(batch_losses['total'], total_loss.item())
                
    # Return mean losses for each output
    return {key: losses.mean() for key, losses in batch_losses.items()}


def add_grad_average_to_tensorboard(writer, model, train_subset_loader, epoch, fold):
    # Log the gradient norms to TensorBoard
    avg_grads = {name: 0 for name, param in model.named_parameters() if param.requires_grad}
    for name, param in model.named_parameters():
        if param.requires_grad:
            avg_grads[name] += param.grad.abs().mean().item()

    # Average over batches and write to tensorboard
    for name, grad_sum in avg_grads.items():
        avg_grad = grad_sum / len(train_subset_loader)
        writer.add_scalar(f'Grad Avg/{name}_fold{fold}', avg_grad, epoch)


def add_learning_rate_to_tensorboard(writer, optimizer, epoch, fold):
    # Log the learning rate to TensorBoard
    for param_group in optimizer.param_groups:
        writer.add_scalar(f'Learning_rate/lr_fold{fold}', param_group['lr'], epoch)


def add_images_to_tensorboard(writer, epoch, fold):
    # Normalize the activations from the 'first_conv_layer'
    images1 = activation['first_conv_layer'][0]

    # Normalize the images to [0,1] range
    images1 = (images1 - images1.min()) / (images1.max() - images1.min())

    # Visualize the first 16 feature maps
    grid1 = make_grid(images1[:16].unsqueeze(1), nrow=4, normalize=False)

    # Resize the grid using interpolation
    grid1 = F.interpolate(grid1.unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0)

    writer.add_image(f'Images/First_layer_fold_{fold}', grid1, epoch)

    # Repeat the same process for the 'second_conv_layer'
    images2 = activation['second_conv_layer'][0]

    # Normalize the images to [0,1] range
    images2 = (images2 - images2.min()) / (images2.max() - images2.min())

    # Visualize the first 16 feature maps
    grid2 = make_grid(images2[:16].unsqueeze(1), nrow=4, normalize=False)

    # Resize the grid using interpolation
    grid2 = F.interpolate(grid2.unsqueeze(0), scale_factor=4, mode='nearest').squeeze(0)

    writer.add_image(f'Images/Second_layer_fold_{fold}', grid2, epoch)


def batch_mean_and_sd():
    loader = dataset_loader.get_full_dataset()
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    print(mean, std)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



class MultiOutputEarlyStopping:
    def __init__(self, patience=5, min_delta=0, monitor='total'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor  # 'total', 'steering', 'throttle', 'brake', or 'weighted_avg'
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, loss_dict):
        # Select which loss to monitor
        if self.monitor == 'total':
            val_loss = loss_dict['total']
        elif self.monitor == 'weighted_avg':
            # Custom weighted average
            val_loss = (
                config.steering_weight * loss_dict['steering'] +
                config.throttle_weight * loss_dict['throttle'] +
                config.brake_weight * loss_dict['brake']
            ) / (config.steering_weight + config.throttle_weight + config.brake_weight)
        else:
            val_loss = loss_dict[self.monitor]
        
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0




def find_corrupted_images(dataset_path):
    csv_path = os.path.join(dataset_path, "steering_data.csv")
    data = pd.read_csv(csv_path)
    
    corrupted_files = []
    valid_indices = []
    
    for idx, row in data.iterrows():
        filename = row['frame_filename']
        camera_pos = row['camera_position']
        img_dir = os.path.join(dataset_path, f"images_{camera_pos}")
        img_path = os.path.join(img_dir, filename)
        
        is_valid = True
        
        # Check if file exists
        if not os.path.exists(img_path):
            corrupted_files.append((idx, img_path, "Missing file"))
            is_valid = False
        else:
            # Test with OpenCV
            try:
                image = cv2.imread(img_path)
                if image is None:
                    corrupted_files.append((idx, img_path, "OpenCV failed"))
                    is_valid = False
            except Exception as e:
                corrupted_files.append((idx, img_path, f"OpenCV error: {e}"))
                is_valid = False
            
            # Test with PIL for PNG-specific errors
            if is_valid:
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verify PNG integrity
                except Exception as e:
                    corrupted_files.append((idx, img_path, f"PIL error: {e}"))
                    is_valid = False
        
        if is_valid:
            valid_indices.append(idx)
    
    return corrupted_files, valid_indices


def create_clean_dataset(dataset_path, valid_indices):
    csv_path = os.path.join(dataset_path, "steering_data.csv")
    data = pd.read_csv(csv_path)
    
    # Keep only valid indices
    clean_data = data.iloc[valid_indices].reset_index(drop=True)
    
    # Save clean CSV
    clean_csv_path = os.path.join(dataset_path, "steering_data_clean.csv")
    clean_data.to_csv(clean_csv_path, index=False)
    
    print(f"Original dataset: {len(data)} samples")
    print(f"Clean dataset: {len(clean_data)} samples")
    print(f"Removed: {len(data) - len(clean_data)} corrupted samples")
    print(f"Clean CSV saved to: {clean_csv_path}")

# Run the check
#dataset_path = "dataset_multioutput/dataset_carla_001_Town10HD_Opt"
#corrupted, valid_indices = find_corrupted_images(dataset_path)

#print(f"Found {len(corrupted)} corrupted files out of total")
#print(f"Valid samples: {len(valid_indices)}")

#create_clean_dataset(dataset_path, valid_indices)