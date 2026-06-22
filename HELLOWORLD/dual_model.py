"""
Dual network model: R3D-18 + Small CNN
"""

import torch
import torch.nn as nn
import torchvision.models.video as models
from config import Config


class SmallArmCNN(nn.Module):
    """
    Small CNN network for classifying arm images
    """
    
    def __init__(self, num_classes: int = 30, input_size: int = 192):
        """
        Initialize small CNN network
        
        Args:
            num_classes: Number of classes
            input_size: Input image size (assumed to be square)
        """
        super(SmallArmCNN, self).__init__()
        
        # Simple CNN architecture
        self.features = nn.Sequential(
            # First layer
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 192 -> 96
            
            # Second layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 96 -> 48
            
            # Third layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 48 -> 24
            
            # Fourth layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 24 -> 12
            
            # Fifth layer
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward propagation
        
        Args:
            x: Input tensor (B, C, H, W) or single image (C, H, W)
        
        Returns:
            Output tensor (B, num_classes) or (num_classes,)
        """
        # If single image, add batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        
        return x


class DualNetworkModel(nn.Module):
    """
    Fusion model: R3D-18 + Small CNN
    """
    
    def __init__(
        self,
        num_classes: int = 30,
        r3d_pretrained: bool = True,
        fusion_method: str = 'weighted_sum',
        r3d_weight: float = 0.7,
        arm_weight: float = 0.3
    ):
        """
        Initialize fusion model
        
        Args:
            num_classes: Number of classes
            r3d_pretrained: Whether R3D uses pre-trained weights
            fusion_method: Fusion method ('weighted_sum', 'concat', 'attention')
            r3d_weight: Weight of R3D output (for weighted_sum)
            arm_weight: Weight of arm CNN output (for weighted_sum)
        """
        super(DualNetworkModel, self).__init__()
        
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.r3d_weight = r3d_weight
        self.arm_weight = arm_weight
        
        # R3D-18 network
        self.r3d_model = models.r3d_18(pretrained=r3d_pretrained)
        num_features_r3d = self.r3d_model.fc.in_features
        self.r3d_model.fc = nn.Linear(num_features_r3d, num_classes)
        
        # Small CNN network (arm images)
        self.arm_cnn = SmallArmCNN(num_classes=num_classes)
        
        # Fusion layer (according to fusion method)
        if fusion_method == 'concat':
            # Concatenate features
            self.fusion_fc = nn.Sequential(
                nn.Linear(num_classes * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        elif fusion_method == 'attention':
            # Attention fusion
            self.attention = nn.Sequential(
                nn.Linear(num_classes * 2, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2),
                nn.Softmax(dim=1)
            )
            self.fusion_fc = nn.Sequential(
                nn.Linear(num_classes * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        # weighted_sum doesn't need additional layers
    
    def forward(self, frames, arm_images_list):
        """
        Forward propagation
        
        Args:
            frames: Action frames (B, C, T, H, W)
            arm_images_list: List of arm images, each sample is a list [(C, H, W), ...]
        
        Returns:
            Output tensor (B, num_classes)
        """
        batch_size = frames.size(0)
        
        # R3D forward propagation
        r3d_output = self.r3d_model(frames)  # (B, num_classes)
        
        # Process arm images
        arm_outputs = []
        for i in range(batch_size):
            arm_images = arm_images_list[i]  # List of (C, H, W)
            
            if len(arm_images) == 0:
                # If no arm images, use zero vector
                arm_output = torch.zeros(self.num_classes, device=frames.device)
            else:
                # Forward pass for each arm image
                arm_logits = []
                for arm_img in arm_images:
                    # Add batch dimension
                    if arm_img.dim() == 3:
                        arm_img = arm_img.unsqueeze(0)
                    logit = self.arm_cnn(arm_img)  # (1, num_classes)
                    arm_logits.append(logit.squeeze(0))  # (num_classes,)
                
                # Average outputs of all arm images
                arm_logits_tensor = torch.stack(arm_logits, dim=0)  # (N, num_classes)
                arm_output = arm_logits_tensor.mean(dim=0)  # (num_classes,)
            
            arm_outputs.append(arm_output)
        
        arm_outputs_tensor = torch.stack(arm_outputs, dim=0)  # (B, num_classes)
        
        # Fuse outputs from two networks
        if self.fusion_method == 'weighted_sum':
            # Weighted sum
            final_output = self.r3d_weight * r3d_output + self.arm_weight * arm_outputs_tensor
        elif self.fusion_method == 'concat':
            # Concatenate then pass through fully connected layer
            concat_features = torch.cat([r3d_output, arm_outputs_tensor], dim=1)  # (B, num_classes*2)
            final_output = self.fusion_fc(concat_features)
        elif self.fusion_method == 'attention':
            # Attention fusion
            concat_features = torch.cat([r3d_output, arm_outputs_tensor], dim=1)  # (B, num_classes*2)
            attention_weights = self.attention(concat_features)  # (B, 2)
            weighted_r3d = attention_weights[:, 0:1] * r3d_output
            weighted_arm = attention_weights[:, 1:2] * arm_outputs_tensor
            weighted_features = torch.cat([weighted_r3d, weighted_arm], dim=1)  # (B, num_classes*2)
            final_output = self.fusion_fc(weighted_features)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        return final_output
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        r3d_params = sum(p.numel() for p in self.r3d_model.parameters())
        arm_cnn_params = sum(p.numel() for p in self.arm_cnn.parameters())
        
        return {
            'model_name': 'DualNetwork (R3D-18 + SmallArmCNN)',
            'num_classes': self.num_classes,
            'fusion_method': self.fusion_method,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'r3d_params': r3d_params,
            'arm_cnn_params': arm_cnn_params
        }


def get_dual_model(
    num_classes: int = None,
    r3d_pretrained: bool = True,
    fusion_method: str = None,
    r3d_weight: float = None,
    arm_weight: float = None
) -> DualNetworkModel:
    """
    Get fusion model
    
    Args:
        num_classes: Number of classes
        r3d_pretrained: Whether R3D uses pre-trained weights
        fusion_method: Fusion method
        r3d_weight: R3D weight
        arm_weight: Arm CNN weight
    
    Returns:
        Fusion model instance
    """
    if num_classes is None:
        num_classes = Config.NUM_CLASSES
    if fusion_method is None:
        fusion_method = Config.FUSION_METHOD
    if r3d_weight is None:
        r3d_weight = Config.R3D_WEIGHT
    if arm_weight is None:
        arm_weight = Config.ARM_WEIGHT
    
    model = DualNetworkModel(
        num_classes=num_classes,
        r3d_pretrained=r3d_pretrained,
        fusion_method=fusion_method,
        r3d_weight=r3d_weight,
        arm_weight=arm_weight
    )
    
    return model


def load_dual_model(
    model_path: str,
    device: str = 'cuda',
    num_classes: int = None,
    fusion_method: str = None,
    r3d_weight: float = None,
    arm_weight: float = None
) -> DualNetworkModel:
    """
    Load trained fusion model
    
    Args:
        model_path: Model file path
        device: Device
        num_classes: Number of classes
        fusion_method: Fusion method
    
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    
    if num_classes is None:
        num_classes = checkpoint_config.get('num_classes', Config.NUM_CLASSES)
    if fusion_method is None:
        fusion_method = checkpoint_config.get('fusion_method', Config.FUSION_METHOD)
    if r3d_weight is None:
        r3d_weight = checkpoint_config.get('r3d_weight', Config.R3D_WEIGHT)
    if arm_weight is None:
        arm_weight = checkpoint_config.get('arm_weight', Config.ARM_WEIGHT)
    
    model = DualNetworkModel(
        num_classes=num_classes,
        r3d_pretrained=False,
        fusion_method=fusion_method,
        r3d_weight=r3d_weight,
        arm_weight=arm_weight
    )
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

