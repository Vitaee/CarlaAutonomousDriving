import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


activation = {}  # initialize the activation dictionary

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class NvidiaModelTransferLearning(nn.Module):
    def __init__(self, pretrained=True, freeze_features=False):
        super().__init__()

        # EfficientNet backbone
        weights =  EfficientNet_B0_Weights.IMAGENET1K_V1
        efficientnet = efficientnet_b0(weights=weights)

        feature_dim = 1280 # EfficientNet-B0 output dimension
        
        # Remove classifier
        self.conv_layers = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        
        # Freeze feature layers if specified
        if freeze_features:
            for param in self.conv_layers.parameters():
                param.requires_grad = False
        
        # Steering prediction head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = self.regressor(x)
        return x.squeeze()
    



class NvidiaModel(nn.Module):
    def __init__(self, pretrained=True, freeze_features=False):
        super().__init__()

        # Load pretrained ResNet34 (lighter than ResNet50 for faster training)
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze feature layers if specified
        if freeze_features:
            for param in self.conv_layers.parameters():
                param.requires_grad = False
        
        # Custom regression head for steering prediction
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),

            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(50, 10),
            nn.ReLU(),
            
            nn.Linear(10, 1)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.regressor(x)
        return x.squeeze()