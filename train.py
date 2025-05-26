
import time
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import dataset_loader as dataset_loader_module
import argparse
from model import NvidiaMultiOutputModel
from config import config
from utils import  train, validation, save_model, MultiOutputEarlyStopping

# Carla/Maps/Nav/Town10HD_Opt.bin
parser = argparse.ArgumentParser(description="Compare loss values from two CSV files.")
parser.add_argument("--dataset_type", type=str, help="Dataset type", choices=['sully', 'udacity', 'udacity_sim_1', 'udacity_sim_2'], default='sully')
parser.add_argument("--batch_size", type=int, help="Batch size", default=config.batch_size)
parser.add_argument("--epochs_count", type=int, help="Epochs count", default=config.epochs_count)
parser.add_argument("--tensorboard_run_name", type=str, help="Tensorboard run name", default='tensorboard')
parser.add_argument("--device", type=str, help="GPU device", default=None)


def main():
    args = parser.parse_args()
    start_time = time.time()

    print(f"Starting Training for:")
    print(f"    TensorBoard Run Name: {args.tensorboard_run_name}")
    print(f"    Number of Epochs: {args.epochs_count}")
    print(f"    Batch Size: {config.batch_size}")
    print(f"    Learning Rate: {config.learning_rate}")
    print(f"    Weight Decay: {config.weight_decay}")
    print(f"    Optimizer: {config.optimizer}")
    print(f"    Number of workers: {config.num_workers}")
    print(f"    Scheduler: {config.scheduler_type}")
    
    if args.device is not None:
        config.device = args.device
    
    # Initialize the TensorBoard writer
    writer = SummaryWriter(log_dir=f'./logs/{args.tensorboard_run_name}/')

    dataset_type = ["carla_001"]
    print("Loading datasets: ", dataset_type)

    train_subset_loader, val_subset_loader = dataset_loader_module.get_data_subsets_loaders(
        dataset_type,
        batch_size=args.batch_size,
        train_only_center=False
    )

    print("Total data size: ", len(train_subset_loader.dataset))

    # Load model
    model = NvidiaMultiOutputModel()
    model.to(config.device)
    
    # Define separate loss functions
    loss_functions = {
        'steering': nn.MSELoss(),
        'throttle': nn.MSELoss(),
        'brake': nn.MSELoss()
    }

    # Optimizer setup
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Invalid optimizer: {config.optimizer}")
    
    # Scheduler setup
    if config.scheduler_type == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    elif config.scheduler_type == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.scheduler_multistep_milestones, gamma=config.scheduler_gamma)
    elif config.scheduler_type == 'nonscheduler':
        scheduler = None
    else:
        raise ValueError(f"Invalid scheduler type: {config.scheduler_type}")

    # Initialize early stopping for multi-output (RECOMMENDED APPROACH)
    early_stopping_train = MultiOutputEarlyStopping(
        patience=config.early_stopping_patience, 
        min_delta=config.early_stopping_min_delta,
        monitor='total'  # Monitor total loss
    )
    early_stopping_val = MultiOutputEarlyStopping(
        patience=config.early_stopping_patience, 
        min_delta=config.early_stopping_min_delta,
        monitor='total'  # Monitor total loss
    )

    
    print("Model and optimizer initialized.")
    print("Starting training...")

    # Start epochs
    for epoch in range(1, args.epochs_count + 1):
        header = f"Epoch: {epoch}/{args.epochs_count}"

        # Train the model      
        epoch_train_losses = train(f"{header}, Training", model, train_subset_loader, loss_functions, optimizer)
        
        # Validate
        epoch_val_losses = validation(f"{header}, Validation", model, val_subset_loader, loss_functions)
        
        
        # Print individual losses
        print(f'{header}, Train - Steering: {epoch_train_losses["steering"]:.6f}, '
              f'Throttle: {epoch_train_losses["throttle"]:.6f}, '
              f'Brake: {epoch_train_losses["brake"]:.6f}, '
              f'Total: {epoch_train_losses["total"]:.6f}')
        
        print()
        print()
        
        print(f'{header}, Val - Steering: {epoch_val_losses["steering"]:.6f}, '
              f'Throttle: {epoch_val_losses["throttle"]:.6f}, '
              f'Brake: {epoch_val_losses["brake"]:.6f}, '
              f'Total: {epoch_val_losses["total"]:.6f}')
        
        print()
        
        # TensorBoard logging
        if config.is_loss_logging_enabled:
            for control_type in ['steering', 'throttle', 'brake', 'total']:
                writer.add_scalars(
                    f'Loss/{control_type}', 
                    {'train': epoch_train_losses[control_type], 'val': epoch_val_losses[control_type]}, 
                    epoch
                )
        
        # Update the learning rate
        if scheduler is not None:
            scheduler.step()

        # Early stopping - pass the loss dictionaries
        early_stopping_train(epoch_train_losses)
        if early_stopping_train.early_stop:
            print(f"Early stopping triggered after {config.early_stopping_patience} epochs without improvement in training loss")
            break
        
        early_stopping_val(epoch_val_losses)
        if early_stopping_val.early_stop:
            print(f"Early stopping triggered after {config.early_stopping_patience} epochs without improvement in validation loss")
            break

    # Save the final model
    save_model(model)

    # Close the TensorBoard writer
    writer.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training took {elapsed_time:.2f} seconds")
    print("Training finished")


if __name__ == '__main__':
    main()
