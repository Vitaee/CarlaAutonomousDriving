import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from typing import Tuple, Dict, Optional
import numpy as np

class SafetyModule(nn.Module):
    """
    Critical safety module for real-world deployment
    Implements uncertainty estimation and anomaly detection
    """
    def __init__(self, input_dim: int):
        super(SafetyModule, self).__init__()
        self.uncertainty_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Softplus()  # Ensures positive uncertainty values
        )
        
        # Anomaly detection - reconstruction loss
        self.anomaly_detector = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        uncertainty = self.uncertainty_head(features)
        reconstruction = self.anomaly_detector(features)
        anomaly_score = F.mse_loss(reconstruction, features, reduction='none').mean(dim=1, keepdim=True)
        
        return uncertainty, anomaly_score

class AttentionFusion(nn.Module):
    """
    Advanced attention mechanism for multi-camera fusion
    """
    def __init__(self, feature_dim: int):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=8,  # REDUCED from 16 for faster processing
            dropout=0.05,  # REDUCED from 0.1
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.positional_encoding = nn.Parameter(torch.randn(3, feature_dim))
        
    def forward(self, center_feat: torch.Tensor, left_feat: torch.Tensor, right_feat: torch.Tensor) -> torch.Tensor:
        batch_size = center_feat.size(0)
        
        # Stack features: [center, left, right]
        features = torch.stack([center_feat, left_feat, right_feat], dim=1)  # (B, 3, feature_dim)
        
        # Add positional encoding for camera positions
        pos_encoding = self.positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        features = features + pos_encoding
        
        # Self-attention across cameras
        attended_features, attention_weights = self.attention(features, features, features)
        attended_features = self.norm(attended_features + features)  # Residual connection
        
        # Weighted fusion based on attention
        fused_features = torch.mean(attended_features, dim=1)  # (B, feature_dim)
        
        return fused_features

class AdvancedLiDARProcessor(nn.Module):
    """
    Sophisticated LiDAR processing with point cloud convolutions
    """
    def __init__(self, lidar_input_size: int = 1000):  # REDUCED from 2000 to match config
        super(AdvancedLiDARProcessor, self).__init__()
        
        # Point cloud feature extraction
        self.point_conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=1),  # REDUCED from 64
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=1),  # REDUCED from 128
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),  # REDUCED from 256
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Linear(128, 256),  # REDUCED from 256→512
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # REDUCED from 0.3
            nn.Linear(256, 256)  # Keep final output size
        )
    
    def forward(self, lidar_points: torch.Tensor) -> torch.Tensor:
        # lidar_points: (B, N, 3) where N is number of points
        batch_size, num_points, _ = lidar_points.shape
        
        # Transpose for conv1d: (B, 3, N)
        points = lidar_points.transpose(1, 2)
        
        # Extract point features
        point_features = self.point_conv(points)  # (B, 256, N)
        
        # Global aggregation
        global_features = self.global_pool(point_features).squeeze(-1)  # (B, 256)
        
        # Refine features
        refined_features = self.feature_refine(global_features)
        
        return refined_features

class ProductionCarlaModel(nn.Module):
    """
    Production-ready autonomous steering model for real vehicle deployment
    
    Features:
    - EfficientNet-B3 backbone for superior visual understanding
    - Advanced multi-camera attention fusion
    - Sophisticated LiDAR processing
    - Uncertainty estimation for safety
    - Anomaly detection
    - Temporal consistency
    - Multiple output heads for different scenarios
    """
    
    def __init__(self, lidar_input_size: int = 1000, num_history_frames: int = 5):
        super(ProductionCarlaModel, self).__init__()
        
        self.num_history_frames = num_history_frames
        
        # EfficientNet-B3 backbone for each camera (REDUCED from B7 for VRAM efficiency)
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        # Remove classifier, keep feature extractor
        self.backbone.classifier = nn.Identity()
        feature_dim = 1536  # EfficientNet-B3 output dimension (was 2560 for B7)
        
        # Freeze early layers for stability (unfreeze during fine-tuning)
        for param in list(self.backbone.parameters())[:50]:  # REDUCED from 100
            param.requires_grad = False
            
        # Advanced camera fusion
        self.camera_fusion = AttentionFusion(feature_dim)
        
        # Advanced LiDAR processing
        self.lidar_processor = AdvancedLiDARProcessor(lidar_input_size)
        
        # Temporal consistency - LSTM for frame history
        self.temporal_lstm = nn.LSTM(
            input_size=feature_dim + 256,  # Camera + LiDAR features
            hidden_size=256,  # REDUCED from 512 for memory efficiency
            num_layers=1,     # REDUCED from 2 for memory efficiency
            batch_first=True
        )
        
        # Feature fusion and processing
        total_features = 256  # LSTM output
        self.feature_processor = nn.Sequential(
            nn.Linear(total_features, 256),  # Keep same size for efficiency
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # REDUCED from 0.3
            nn.Linear(256, 256)  # Single layer for speed
        )
        
        # Multiple output heads for different scenarios
        self.steering_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()  # Normalized steering [-1, 1]
        )
        
        # Speed prediction head (for adaptive behavior)
        self.speed_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Normalized speed [0, 1]
        )
        
        # Safety-critical components
        self.safety_module = SafetyModule(256)
        
        # Emergency brake classifier
        self.emergency_brake = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),  # [no_brake, emergency_brake]
            nn.Softmax(dim=1)
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize custom layers with proper weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def extract_camera_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from camera images using EfficientNet-B3 with gradient checkpointing"""
        if self.training:
            # Use gradient checkpointing during training to save memory
            features = torch.utils.checkpoint.checkpoint(self.backbone, images, use_reentrant=False)
        else:
            features = self.backbone(images)
        return features
        
    def forward(self, 
                center_imgs: torch.Tensor, 
                left_imgs: torch.Tensor, 
                right_imgs: torch.Tensor, 
                lidar_points: torch.Tensor,
                vehicle_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with comprehensive outputs for safety
        
        Args:
            center_imgs: (B, T, 3, H, W) - Temporal sequence of center camera
            left_imgs: (B, T, 3, H, W) - Temporal sequence of left camera  
            right_imgs: (B, T, 3, H, W) - Temporal sequence of right camera
            lidar_points: (B, T, N, 3) - Temporal sequence of LiDAR points
            vehicle_state: (B, T, state_dim) - Optional vehicle state info
            
        Returns:
            Dictionary containing all predictions and safety metrics
        """
        batch_size, seq_len = center_imgs.size(0), center_imgs.size(1)
        
        # Process each time step
        temporal_features = []
        
        for t in range(seq_len):
            # Extract camera features for this timestep
            center_feat = self.extract_camera_features(center_imgs[:, t])
            left_feat = self.extract_camera_features(left_imgs[:, t])
            right_feat = self.extract_camera_features(right_imgs[:, t])
            
            # Fuse camera features with attention
            fused_camera_feat = self.camera_fusion(center_feat, left_feat, right_feat)
            
            # Process LiDAR for this timestep
            lidar_feat = self.lidar_processor(lidar_points[:, t])
            
            # Combine camera and LiDAR features
            combined_feat = torch.cat([fused_camera_feat, lidar_feat], dim=1)
            temporal_features.append(combined_feat)
        
        # Stack temporal features
        temporal_features = torch.stack(temporal_features, dim=1)  # (B, T, feature_dim)
        
        # Process through LSTM for temporal consistency
        lstm_out, (hidden, cell) = self.temporal_lstm(temporal_features)
        
        # Use the last timestep output
        final_features = lstm_out[:, -1, :]  # (B, 256)
        
        # Process features
        processed_features = self.feature_processor(final_features)
        
        # Generate all predictions
        steering = self.steering_head(processed_features).squeeze(-1)  # (B,)
        speed = self.speed_head(processed_features).squeeze(-1)        # (B,)
        emergency_brake_prob = self.emergency_brake(processed_features)  # (B, 2)
        
        # Safety assessments
        uncertainty, anomaly_score = self.safety_module(processed_features)
        
        return {
            'steering': steering,
            'speed': speed,
            'emergency_brake': emergency_brake_prob,
            'uncertainty': uncertainty,
            'anomaly_score': anomaly_score,
            'features': processed_features  # For debugging/analysis
        }
    
    def predict_safe(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Safe prediction with validation checks
        """
        with torch.no_grad():
            outputs = self.forward(*args, **kwargs)
            
            # Safety checks
            uncertainty_threshold = 0.5
            anomaly_threshold = 0.3
            
            # Flag high uncertainty or anomaly detections
            high_uncertainty = outputs['uncertainty'] > uncertainty_threshold
            high_anomaly = outputs['anomaly_score'] > anomaly_threshold
            
            # Conservative steering when uncertain
            if high_uncertainty.any() or high_anomaly.any():
                outputs['steering'] = outputs['steering'] * 0.5  # Reduce steering magnitude
                outputs['safety_warning'] = True
            else:
                outputs['safety_warning'] = False
                
            return outputs

class ModelTrainer:
    """
    Specialized trainer for the production model with safety considerations
    """
    def __init__(self, model: ProductionCarlaModel, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Multi-task loss components
        self.steering_criterion = nn.MSELoss()
        self.speed_criterion = nn.MSELoss()
        self.brake_criterion = nn.CrossEntropyLoss()
        self.uncertainty_criterion = nn.MSELoss()
        
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute multi-task loss with proper weighting
        """
        # Primary task losses
        steering_loss = self.steering_criterion(outputs['steering'], targets['steering'])
        speed_loss = self.speed_criterion(outputs['speed'], targets['speed'])
        brake_loss = self.brake_criterion(outputs['emergency_brake'], targets['emergency_brake'])
        
        # Uncertainty loss (epistemic uncertainty)
        uncertainty_target = torch.abs(outputs['steering'] - targets['steering']).unsqueeze(-1)  # (B, 1)
        uncertainty_loss = self.uncertainty_criterion(outputs['uncertainty'], uncertainty_target)
        
        # Anomaly detection loss
        anomaly_loss = torch.mean(outputs['anomaly_score'])
        
        # Weighted combination
        total_loss = (2.0 * steering_loss +      # Primary task
                     1.0 * speed_loss +          # Secondary task
                     1.5 * brake_loss +          # Safety critical
                     0.5 * uncertainty_loss +    # Uncertainty estimation
                     0.3 * anomaly_loss)         # Anomaly detection
        
        return total_loss

# Example usage and testing
if __name__ == "__main__":
    # Initialize production model with optimized settings
    model = ProductionCarlaModel(lidar_input_size=1000, num_history_frames=5)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test with dummy data matching new config
    batch_size, seq_len = 2, 5
    height, width = 192, 192  # Updated to match config image_size
    num_lidar_points = 500  # Reduced for testing
    
    # Create dummy inputs
    center_imgs = torch.randn(batch_size, seq_len, 3, height, width)
    left_imgs = torch.randn(batch_size, seq_len, 3, height, width)
    right_imgs = torch.randn(batch_size, seq_len, 3, height, width)
    lidar_points = torch.randn(batch_size, seq_len, num_lidar_points, 3)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(center_imgs, left_imgs, right_imgs, lidar_points)
    
    print("\nModel Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
    
    print(f"\nSteering predictions: {outputs['steering'].flatten()}")
    print(f"Uncertainty estimates: {outputs['uncertainty'].flatten()}")
    print(f"Anomaly scores: {outputs['anomaly_score'].flatten()}")
    print(f"\nMemory-optimized model ready for RTX 4070!")