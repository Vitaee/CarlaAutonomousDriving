from torchvision.utils import make_grid
import torch.nn.functional as F
import torch
import dataset_loader
import numpy as np
from tqdm import tqdm
import os
from config import config
from model import activation


def save_model(model, log_dir="./save_center"):
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


def train(desc_message, model, train_subset_loader, loss_function, optimizer):
    model.train()
    batch_loss = np.array([])

    for data, target in tqdm(train_subset_loader, desc=desc_message, ascii=' ='):

        data = data.to(config.device)
        target = target.to(config.device)

        optimizer.zero_grad()

        y_pred = model(data)
        
        loss = loss_function(y_pred.float(), target.float())
    
        loss.backward()
        optimizer.step()

        batch_loss = np.append(batch_loss, [loss.item()])

        epoch_loss = batch_loss.mean()

    return epoch_loss


def validation(desc_message, model, val_subset_loader, loss_function):
    # Load model
    model.eval()
    batch_loss = np.array([])

    with torch.no_grad():
        for data_val, target_val in tqdm(val_subset_loader, desc=desc_message, ascii=' ='):
            # send data to device (its is medatory if GPU has to be used)
            data_val = data_val.to(config.device)

            # send target to device
            target_val = target_val.to(config.device)
            
            # forward pass to the model
            y_pred_val = model(data_val)

            # cross entropy loss
            loss = loss_function(y_pred_val.float(), target_val.float())

            # Capture log
            batch_loss = np.append(batch_loss, [loss.item()])
                
    epoch_loss = batch_loss.mean()

    return epoch_loss


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
