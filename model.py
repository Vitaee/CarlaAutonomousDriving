import torch
import torch.nn as nn
from config import config

activation = {}  # initialize the activation dictionary


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook





class NvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()

        # define layers using nn.Sequential
        self.conv_layers = nn.Sequential(
            # first convolutional layer
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            # second convolutional layer
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),

            # third convolutional layer
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # fourth convolutional layer
            nn.Conv2d(48, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # fifth convolutional layer
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        if config.is_image_logging_enabled:
            self.conv_layers[2].register_forward_hook(get_activation('first_conv_layer'))
            self.conv_layers[5].register_forward_hook(get_activation('second_conv_layer'))
        
        self.flat_layers = nn.Sequential(
            # flatten
            nn.Flatten(),
            nn.Dropout(p=0.5),
            
            # first fully connected layer
            nn.Linear(1152, 1164),
            nn.BatchNorm1d(1164),
            nn.ReLU(),
            
            # second fully connected layer
            nn.Linear(1164, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),

            # third fully connected layer
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),

            # fourth fully connected layer
            nn.Linear(50, 10),

            # output layer
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flat_layers(x)
        return x.squeeze()


class NvidiaModelTransferLearning(nn.Module):
    def __init__(self, resnet):
        super().__init__()

        # Use the pretrained ResNet model as the convolutional layers
        self.conv_layers = resnet

        # Define the flat layers as before
        self.flat_layers = nn.Sequential(
            # flatten
            nn.Flatten(),
            nn.Dropout(p=0.5),

            # first fully connected layer
            nn.Linear(512, 1164),
            nn.BatchNorm1d(1164),
            nn.ReLU(),

            # second fully connected layer
            nn.Linear(1164, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),

            # third fully connected layer
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),

            # fourth fully connected layer
            nn.Linear(50, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),

            # output layer
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flat_layers(x)
        return x.squeeze()


class NvidiaMultiOutputModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared convolutional layers (same as before)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(1152, 1164),
            nn.BatchNorm1d(1164),
            nn.ReLU(),
            nn.Linear(1164, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
        )

        # Separate output heads for each control
        self.steering_head = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Tanh()  # Steering: [-1, 1]
        )
        
        self.throttle_head = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()  # Throttle: [0, 1]
        )
        
        self.brake_head = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()  # Brake: [0, 1]
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.shared_fc(x)
        
        steering = self.steering_head(x).squeeze()
        throttle = self.throttle_head(x).squeeze()
        brake = self.brake_head(x).squeeze()
        
        return {
            'steering': steering,
            'throttle': throttle,
            'brake': brake
        }