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
    def __init__(self, pretrained=True, freeze_features=False, use_speed_input=False):
        super().__init__()
        
        self.use_speed_input = use_speed_input

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
        
        # Speed embedding (if using speed input)
        if self.use_speed_input:
            # Speed normalization and embedding
            self.speed_embedding = nn.Sequential(
                nn.Linear(1, 16),  # Speed input (1D) -> 16D embedding
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            # Combine visual features + speed features
            combined_dim = feature_dim + 32
        else:
            combined_dim = feature_dim
        
        # Steering prediction head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            
            nn.Linear(combined_dim, 256),
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
    
    def forward(self, x, speed=None):
        # Extract visual features
        visual_features = self.conv_layers(x)
        visual_features = self.avgpool(visual_features)
        visual_features = torch.flatten(visual_features, 1)
        
        if self.use_speed_input and speed is not None:
            # Normalize speed (expected range: 0-100 km/h)
            speed_normalized = speed / 100.0
            speed_features = self.speed_embedding(speed_normalized.unsqueeze(-1))
            
            # Concatenate visual and speed features
            combined_features = torch.cat([visual_features, speed_features], dim=1)
        else:
            combined_features = visual_features
        
        # Predict steering
        steering = self.regressor(combined_features)
        return steering.squeeze()
    



class NvidiaModel(nn.Module):
    def __init__(self, pretrained=True, freeze_features=False, use_speed_input=False):
        super().__init__()
        
        self.use_speed_input = use_speed_input

        # Load pretrained ResNet34 (lighter than ResNet50 for faster training)
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze feature layers if specified
        if freeze_features:
            for param in self.conv_layers.parameters():
                param.requires_grad = False
        
        # Speed embedding (if using speed input)
        if self.use_speed_input:
            self.speed_embedding = nn.Sequential(
                nn.Linear(1, 16),  # Speed input (1D) -> 16D embedding
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            # Combine ResNet features (2048) + speed features (32)
            combined_dim = 2048 + 32
        else:
            combined_dim = 2048
        
        # Custom regression head for steering prediction
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),

            nn.Linear(combined_dim, 256),
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
        
    def forward(self, x, speed=None):
        # Extract visual features
        visual_features = self.conv_layers(x)
        visual_features = torch.flatten(visual_features, 1)
        
        if self.use_speed_input and speed is not None:
            # Normalize speed (expected range: 0-100 km/h)
            speed_normalized = speed / 100.0
            speed_features = self.speed_embedding(speed_normalized.unsqueeze(-1))
            
            # Concatenate visual and speed features
            combined_features = torch.cat([visual_features, speed_features], dim=1)
        else:
            combined_features = visual_features
        
        # Predict steering
        steering = self.regressor(combined_features)
        return steering.squeeze()