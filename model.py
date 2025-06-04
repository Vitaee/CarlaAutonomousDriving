import torch
import torch.nn as nn
import torchvision.models as models


activation = {}  # initialize the activation dictionary

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


class MultiControlAutonomousModel(nn.Module):
    """
    🚗 ENHANCED MULTI-OUTPUT AUTONOMOUS DRIVING MODEL
    
    Outputs:
    - Steering angle (radians)
    - Throttle (0-1)
    - Brake (0-1)
    
    Optional Speed Input:
    - Current speed can be provided as additional input for better control decisions
    """
    
    def __init__(self, pretrained=True, freeze_features=False, use_speed_input=True):
        super().__init__()
        
        self.use_speed_input = use_speed_input
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer  
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze feature layers if specified
        if freeze_features:
            for param in self.conv_layers.parameters():
                param.requires_grad = False
        
        # Feature fusion layer (combines visual features with speed if used)
        feature_input_size = 2048 + (1 if use_speed_input else 0)
        
        # Shared feature processing
        self.shared_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        
        # Steering Head - Most complex (requires precise turning)
        self.steering_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)  # Steering angle (-1 to 1, then converted to radians)
        )
        
        # Throttle Head - Smooth acceleration control
        self.throttle_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 range
        )
        
        # Brake Head - Safety-critical
        self.brake_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 range
        )
        
    def forward(self, x, current_speed=None):
        # Extract visual features
        visual_features = self.conv_layers(x)  # [batch, 2048, 1, 1]
        visual_features = visual_features.view(visual_features.size(0), -1)  # [batch, 2048]
        
        # Optionally fuse with speed information
        if self.use_speed_input and current_speed is not None:
            # Normalize speed (assuming max speed ~100 km/h)
            normalized_speed = current_speed.unsqueeze(1) / 100.0  # [batch, 1]
            combined_features = torch.cat([visual_features, normalized_speed], dim=1)
        else:
            combined_features = visual_features
        
        # Process through shared layers
        shared_features = self.shared_layers(combined_features)  # [batch, 256]
        
        # Generate control outputs
        steering = self.steering_head(shared_features)  # [batch, 1]
        throttle = self.throttle_head(shared_features)  # [batch, 1] 
        brake = self.brake_head(shared_features)       # [batch, 1]
        
        return {
            'steering': steering.squeeze(),    # Remove extra dimension
            'throttle': throttle.squeeze(),
            'brake': brake.squeeze()
        }


class SpeedAwareAutonomousModel(nn.Module):
    """
    🎯 SPEED-AWARE MODEL WITH TARGET SPEED CONTROL
    
    This model can take a target speed as input and learn to control
    throttle/brake to maintain that speed while steering appropriately.
    """
    
    def __init__(self, pretrained=True, freeze_features=False):
        super().__init__()
        
        # Visual feature extractor
        resnet = models.resnet50(pretrained=pretrained)
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze_features:
            for param in self.conv_layers.parameters():
                param.requires_grad = False
        
        # Input: visual features (2048) + current speed (1) + target speed (1)
        self.feature_fusion = nn.Sequential(
            nn.Linear(2048 + 2, 512),  # visual + current_speed + target_speed
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Steering head (independent of speed)
        self.steering_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Speed control head (throttle/brake coordination)
        self.speed_control_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [throttle, brake]
            nn.Sigmoid()
        )
        
    def forward(self, x, current_speed, target_speed):
        # Extract visual features
        visual_features = self.conv_layers(x).view(x.size(0), -1)
        
        # Normalize speeds
        current_speed_norm = current_speed.unsqueeze(1) / 100.0
        target_speed_norm = target_speed.unsqueeze(1) / 100.0
        
        # Combine all inputs
        combined = torch.cat([visual_features, current_speed_norm, target_speed_norm], dim=1)
        
        # Process features
        features = self.feature_fusion(combined)
        
        # Generate outputs
        steering = self.steering_head(features)
        speed_controls = self.speed_control_head(features)
        
        return {
            'steering': steering.squeeze(),
            'throttle': speed_controls[:, 0],
            'brake': speed_controls[:, 1]
        }