"""
Video dataset class
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
import utils
from config import Config

class VideoDataset(Dataset):
    """
    Video dataset class
    """
    def __init__(
        self,
        csv_file: str,
        video_dir: str,
        num_frames: int = 16,
        image_size: Tuple[int, int] = (112, 112),
        is_training: bool = True,
        class_ind_file: Optional[str] = None
    ):
        """
        Initialize dataset
        
        Args:
            csv_file: CSV file path containing video IDs and labels
            video_dir: Video file directory
            num_frames: Number of frames sampled per video
            image_size: Image size (height, width)
            is_training: Whether in training mode (affects data augmentation)
            class_ind_file: Class index file path
        """
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.image_size = image_size
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
        
        # Ensure class IDs start from 0 (if needed)
        self.num_classes = len(self.idx_to_class)
        
        print(f"Loaded dataset: {len(self.data)} samples, {self.num_classes} classes")
        print(f"Training mode: {is_training}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample
        
        Args:
            idx: Sample index
        
        Returns:
            (frames, label): Frame tensor and label
        """
        row = self.data.iloc[idx]
        video_id = row['video_id']
        class_id = row['class_id']
        
        # Build video file path
        video_path = os.path.join(self.video_dir, f"{video_id}.avi")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        
        # Sample frames
        if self.is_training and Config.TRAIN_TEMPORAL_JITTER:
            mode = 'random'
        else:
            mode = 'uniform'
        
        frames = utils.sample_frames(video_path, self.num_frames, mode=mode)
        
        # Data augmentation
        processed_frames = []
        for frame in frames:
            # Resize or crop
            if self.is_training:
                if Config.TRAIN_RANDOM_CROP:
                    frame = utils.random_crop(frame, self.image_size)
                else:
                    frame = utils.resize_frame(frame, self.image_size)
                
                # Horizontal flip
                if Config.TRAIN_HORIZONTAL_FLIP and np.random.random() < 0.5:
                    frame = utils.horizontal_flip(frame)
            else:
                if Config.VAL_CENTER_CROP:
                    frame = utils.center_crop(frame, self.image_size)
                else:
                    frame = utils.resize_frame(frame, self.image_size)
            
            processed_frames.append(frame)
        
        # Normalization
        frames_array = utils.normalize_frames(processed_frames)
        
        # Convert to tensor (T, C, H, W) -> (C, T, H, W) for 3D CNN
        frames_tensor = torch.from_numpy(frames_array).float()
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        # Label (class ID starts from 1, need to convert to start from 0)
        label = class_id - 1  # Assume class ID starts from 1
        
        return frames_tensor, label
    
    def get_class_names(self) -> dict:
        """Get class names dictionary"""
        return self.idx_to_class.copy()

def create_dataloader(
    csv_file: str,
    video_dir: str,
    batch_size: int = 8,
    num_frames: int = 16,
    image_size: Tuple[int, int] = (112, 112),
    is_training: bool = True,
    num_workers: int = 4,
    class_ind_file: Optional[str] = None,
    shuffle: Optional[bool] = None
) -> torch.utils.data.DataLoader:
    """
    Create data loader
    
    Args:
        csv_file: CSV file path
        video_dir: Video file directory
        batch_size: Batch size
        num_frames: Number of frames per video
        image_size: Image size
        is_training: Whether in training mode
        num_workers: Number of data loading threads
        class_ind_file: Class index file
        shuffle: Whether to shuffle (if None, determined by is_training)
    
    Returns:
        DataLoader instance
    """
    dataset = VideoDataset(
        csv_file=csv_file,
        video_dir=video_dir,
        num_frames=num_frames,
        image_size=image_size,
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
        drop_last=is_training  # Drop last incomplete batch during training
    )
    
    return dataloader

