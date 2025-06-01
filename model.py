import torch
import torch.nn as nn
import torchvision.models as models


class NvidiaModelTransferLearning(nn.Module):
    def __init__(self, pretrained=True, freeze_features=False):
        super().__init__()
        
        # Load pretrained ResNet18 (lighter than ResNet50 for faster training)
        resnet = models.resnet18(pretrained=pretrained)
        
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
            
            nn.Linear(512, 256),  # ResNet18 outputs 512 features
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