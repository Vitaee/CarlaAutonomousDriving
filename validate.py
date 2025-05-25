import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dataset_loader
import numpy as np
import pandas as pd
from model import NvidiaModel
from config import config


def main():
    # train over the dataset about 30 times
    _, val_subset_loader = dataset_loader.get_data_subsets_loaders(dataset_types=['carla_001'], train_only_center=False)
    
    # Load model
    model = NvidiaModel()
    model.load_state_dict(torch.load("./save/model.pt", map_location=torch.device(config.device)))
    model.to(config.device)
    model.eval()
    
    # Loss function using MSE
    loss_function = nn.MSELoss()

    # to get batch loss
    batch_loss = np.array([])
    batch_loss_mean = np.array([])

    for batch_idx, (data, target) in enumerate(val_subset_loader):
        # send data to device (its is medatory if GPU has to be used)
        data = data.to(config.device)

        # send target to device
        target = target.to(config.device)
        
        with torch.no_grad():
            # forward pass to the model
            y_pred = model(data)

            # cross entropy loss
            loss = loss_function(y_pred, target)

        # Capture log
        batch_loss = np.append(batch_loss, [loss.item()])

        if batch_idx % 5 == 0:
            epoch_loss = batch_loss.mean()
            batch_loss_mean = np.append(batch_loss_mean, [epoch_loss])
            print(f'Validation Loss: {epoch_loss:.6f}')

    loss_acc_df = pd.DataFrame({"loss": batch_loss_mean})
    loss_acc_df.to_csv("loss_acc_results_validation.csv", index=None)
    print("loss_acc_results_validation.csv saved!")


if __name__ == '__main__':
    main()
