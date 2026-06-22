"""
Preprocessed dataset class
Load 16 action frames and arm images from preprocessed_frames folder
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from config import Config


def is_black_image(image_path: str, threshold: float = 0.01) -> bool:
    """
    Check if image is pure black placeholder image
    
    Args:
        image_path: Image path
        threshold: Pixel value threshold (if mean of all pixels is less than threshold, considered black)
    
    Returns:
        Whether it is a black image
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        # Calculate mean of all pixels
        mean_value = np.mean(img_array)
        return mean_value < threshold * 255
    except:
        return True


class PreprocessedDataset(Dataset):
    """
    Preprocessed dataset class
    Load data from preprocessed_frames folder
    """
    
    def __init__(
        self,
        csv_file: str,
        preprocessed_dir: str,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (384, 384),
        arm_size: Tuple[int, int] = (192, 192),
        is_training: bool = True,
        class_ind_file: Optional[str] = None
    ):
        """
        Initialize dataset
        
        Args:
            csv_file: CSV file path containing video IDs and labels
            preprocessed_dir: Preprocessed frame folder directory
            num_frames: Number of frames per video (should be 16)
            frame_size: Action frame size (H, W)
            arm_size: Arm image size (H, W)
            is_training: Whether in training mode (affects data augmentation)
            class_ind_file: Class index file path
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.arm_size = arm_size
        self.is_training = is_training
        
        # Load data
        self.data = pd.read_csv(csv_file, header=None, names=['video_id', 'class_name', 'class_id'])
        
        # Load class mapping
        self.class_to_idx = {}
        self.idx_to_class = {}
        if class_ind_file and os.path.exists(class_ind_file):
            with open(class_ind_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            class_id = int(parts[0])
                            class_name = parts[1]
                            self.class_to_idx[class_name] = class_id
                            self.idx_to_class[class_id] = class_name
        else:
            # If no class file, build from data
            unique_classes = sorted(self.data['class_id'].unique())
            for class_id in unique_classes:
                self.idx_to_class[class_id] = f'Class_{class_id}'
                class_name = self.data[self.data['class_id'] == class_id]['class_name'].iloc[0]
                self.class_to_idx[class_name] = class_id
        
        self.num_classes = len(self.idx_to_class)
        
        # Data augmentation
        if is_training:
            self.frame_transform = transforms.Compose([
                transforms.Resize(frame_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                   std=[0.22803, 0.22145, 0.216989])
            ])
            self.arm_transform = transforms.Compose([
                transforms.Resize(arm_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.frame_transform = transforms.Compose([
                transforms.Resize(frame_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                   std=[0.22803, 0.22145, 0.216989])
            ])
            self.arm_transform = transforms.Compose([
                transforms.Resize(arm_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        print(f"Loaded preprocessed dataset: {len(self.data)} samples, {self.num_classes} classes")
        print(f"Training mode: {is_training}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """
        Get a sample
        
        Args:
            idx: Sample index
        
        Returns:
            (frames, arm_images, label): 
            - frames: 16 action frame tensor (C, T, H, W)
            - arm_images: Arm image list (each is (C, H, W))
            - label: Label (0-based)
        """
        row = self.data.iloc[idx]
        video_id = row['video_id']
        class_id = row['class_id']
        
        # Build video folder path
        video_dir = self.preprocessed_dir / video_id
        
        if not video_dir.exists():
            raise FileNotFoundError(f"Video folder does not exist: {video_dir}")
        
        # Load 16 action frames
        frames = []
        for i in range(self.num_frames):
            # Find corresponding frame's person image
            frame_files = sorted(video_dir.glob(f"frame_{i:03d}_person_*.png"))
            if frame_files:
                # If multiple persons, take the first one
                frame_path = frame_files[0]
                img = Image.open(frame_path).convert('RGB')
                img_tensor = self.frame_transform(img)
                frames.append(img_tensor)
            else:
                # If not found, use black placeholder
                placeholder = Image.new('RGB', self.frame_size, (0, 0, 0))
                img_tensor = self.frame_transform(placeholder)
                frames.append(img_tensor)
        
        # Convert to (C, T, H, W)
        frames_tensor = torch.stack(frames, dim=1)  # (C, T, H, W)
        
        # Load arm images (ignore pure black placeholder images)
        arm_images = []
        arm_files = sorted(video_dir.glob("frame_*_arm_*.png"))
        for arm_file in arm_files:
            # Check if it's a black image
            if not is_black_image(str(arm_file)):
                img = Image.open(arm_file).convert('RGB')
                img_tensor = self.arm_transform(img)
                arm_images.append(img_tensor)
        
        # If no valid arm images, create a placeholder (all-zero tensor)
        if len(arm_images) == 0:
            placeholder = Image.new('RGB', self.arm_size, (0, 0, 0))
            arm_images.append(self.arm_transform(placeholder))
        
        # Label (class ID starts from 1, need to convert to start from 0)
        label = class_id - 1
        
        return frames_tensor, arm_images, label
    
    def get_class_names(self) -> dict:
        """Get class names dictionary"""
        return self.idx_to_class.copy()


def collate_fn(batch):
    """
    Custom collate function to handle variable-length arm image lists
    
    Args:
        batch: Batch data list
    
    Returns:
        (frames_batch, arm_images_list, labels_batch):
        - frames_batch: (B, C, T, H, W)
        - arm_images_list: List[List[Tensor]], arm image list for each sample
        - labels_batch: (B,)
    """
    frames_list = []
    arm_images_list = []
    labels_list = []
    
    for frames, arm_images, label in batch:
        frames_list.append(frames)
        arm_images_list.append(arm_images)
        labels_list.append(label)
    
    # Stack frames
    frames_batch = torch.stack(frames_list, dim=0)  # (B, C, T, H, W)
    
    # Labels
    labels_batch = torch.tensor(labels_list, dtype=torch.long)
    
    return frames_batch, arm_images_list, labels_batch


def create_dataloader(
    csv_file: str,
    preprocessed_dir: str,
    batch_size: int = 8,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (384, 384),
    arm_size: Tuple[int, int] = (192, 192),
    is_training: bool = True,
    num_workers: int = 4,
    class_ind_file: Optional[str] = None,
    shuffle: Optional[bool] = None
) -> torch.utils.data.DataLoader:
    """
    Create data loader
    
    Args:
        csv_file: CSV file path
        preprocessed_dir: Preprocessed frame folder directory
        batch_size: Batch size
        num_frames: Number of frames per video
        frame_size: Action frame size
        arm_size: Arm image size
        is_training: Whether in training mode
        num_workers: Number of data loading threads
        class_ind_file: Class index file
        shuffle: Whether to shuffle (if None, determined by is_training)
    
    Returns:
        DataLoader instance
    """
    dataset = PreprocessedDataset(
        csv_file=csv_file,
        preprocessed_dir=preprocessed_dir,
        num_frames=num_frames,
        frame_size=frame_size,
        arm_size=arm_size,
        is_training=is_training,
        class_ind_file=class_ind_file
    )
    
    if shuffle is None:
        shuffle = is_training
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_training,
        collate_fn=collate_fn
    )
    
    return dataloader

