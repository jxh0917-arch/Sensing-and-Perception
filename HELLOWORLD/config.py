"""
Configuration file
Define all hyperparameters, paths, and model configurations
For dual model (R3D-18 + CNN) training and testing
"""

import os
from pathlib import Path

class Config:
    """Configuration class - Dual model (R3D-18 + CNN) configuration"""
    
    # ========== Path Configuration ==========
    # Note: Now using preprocessed data, no longer directly using raw video directories
    # Preprocessed data directories: 'preprocessed_frames' (train/val/test)
    #                                'preprocessed_test_frames' (test set application)
    ANNOTATIONS_DIR = Path('annotations')
    
    # Label files
    CLASS_IND_FILE = ANNOTATIONS_DIR / 'classInd.txt'
    TRAIN_CSV = ANNOTATIONS_DIR / 'train.csv'
    VAL_CSV = ANNOTATIONS_DIR / 'val.csv'
    TEST_CSV = ANNOTATIONS_DIR / 'test.csv'
    
    # Model save paths
    MODEL_DIR = Path('checkpoints')
    BEST_MODEL_PATH = MODEL_DIR / 'best_model.pth'
    LAST_MODEL_PATH = MODEL_DIR / 'last_model.pth'
    
    # Log and results paths
    LOG_DIR = Path('logs')
    RESULTS_DIR = Path('results')
    
    # ========== Model Configuration ==========
    NUM_CLASSES = 30  # Number of classes
    NUM_FRAMES = 16  # Number of frames sampled per video
    # Note: Image sizes are now fixed as:
    # - Action frames: (384, 384) - defined in preprocessed_dataset.py
    # - Arm images: (192, 192) - defined in preprocessed_dataset.py
    
    # R3D-18 configuration (fixed to use r3d_18, no longer need MODEL_NAME parameter)
    PRETRAINED = True  # Whether to use pre-trained weights
    
    # Fusion configuration
    FUSION_METHOD = 'weighted_sum'
    R3D_WEIGHT = 0.7
    ARM_WEIGHT = 0.3
    
    # ========== Training Configuration ==========
    BATCH_SIZE = 2  # Batch size (adjust according to GPU memory)
    NUM_EPOCHS = 100  # Number of training epochs
    NUM_WORKERS = 4  # Number of data loading threads
    
    # Optimizer configuration
    OPTIMIZER = 'Adam'  # 'Adam' or 'SGD'
    LEARNING_RATE = 1e-4  # Initial learning rate
    WEIGHT_DECAY = 1e-4  # Weight decay
    MOMENTUM = 0.9  # SGD momentum (only for SGD)
    
    # Learning rate scheduling
    LR_SCHEDULER = 'CosineAnnealingLR'  # 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'
    LR_STEP_SIZE = 10  # StepLR step size
    LR_GAMMA = 0.1  # Learning rate decay factor
    LR_MIN = 1e-6  # Minimum learning rate
    
    # Loss function
    LOSS_FUNCTION = 'CrossEntropyLoss'
    LABEL_SMOOTHING = 0.0  # Label smoothing (0 means not used)
    
    # Training strategy
    GRADIENT_CLIP = 1.0  # Gradient clipping threshold (0 means no clipping)
    EARLY_STOPPING_PATIENCE = 5  # Early stopping patience value
    
    # Note: Data augmentation is now handled in preprocessed_dataset.py
    # During training: random horizontal flip, color jitter
    # During testing: no augmentation
    
    # ========== Device Configuration ==========
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    CUDA_DEVICE = 0  # GPU device ID
    
    # ========== Other Configuration ==========
    SEED = 42  # Random seed
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        cls.MODEL_DIR.mkdir(exist_ok=True)
        cls.LOG_DIR.mkdir(exist_ok=True)
        cls.RESULTS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_class_names(cls):
        """Load class names"""
        class_names = {}
        if cls.CLASS_IND_FILE.exists():
            with open(cls.CLASS_IND_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            class_id = int(parts[0])
                            class_name = parts[1]
                            class_names[class_id] = class_name
        return class_names
    
    @classmethod
    def get_num_classes(cls):
        """Get number of classes"""
        return cls.NUM_CLASSES

# Create global configuration instance
config = Config()
