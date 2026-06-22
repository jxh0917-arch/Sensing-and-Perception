"""
Dual network training script
Train 16 action frames using R3D-18, train arm images using small CNN
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import time
from tqdm import tqdm
import json
from pathlib import Path

from config import Config
from dual_model import get_dual_model
from preprocessed_dataset import create_dataloader
import utils


def get_optimizer(model, config):
    """Get optimizer"""
    if config.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")
    
    return optimizer


def get_scheduler(optimizer, config, num_batches_per_epoch=None):
    """Get learning rate scheduler"""
    if config.LR_SCHEDULER == 'StepLR':
        scheduler = StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)
    elif config.LR_SCHEDULER == 'CosineAnnealingLR':
        if num_batches_per_epoch:
            T_max = config.NUM_EPOCHS * num_batches_per_epoch
        else:
            T_max = config.NUM_EPOCHS
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=config.LR_MIN)
    elif config.LR_SCHEDULER == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.LR_GAMMA,
            patience=5,
            min_lr=config.LR_MIN
        )
    else:
        scheduler = None
    
    return scheduler


def get_loss_function(config):
    """Get loss function"""
    if config.LOSS_FUNCTION == 'CrossEntropyLoss':
        if config.LABEL_SMOOTHING > 0:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {config.LOSS_FUNCTION}")
    
    return loss_fn


def train_epoch(model, dataloader, optimizer, loss_fn, device, config, epoch):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]')
    
    for batch_idx, (frames, arm_images_list, labels) in enumerate(pbar):
        frames = frames.to(device)
        labels = labels.to(device)
        
        # Move each tensor in arm_images_list to device
        arm_images_list_device = []
        for arm_images in arm_images_list:
            arm_images_device = [img.to(device) for img in arm_images]
            arm_images_list_device.append(arm_images_device)
        
        # Forward propagation
        optimizer.zero_grad()
        outputs = model(frames, arm_images_list_device)
        loss = loss_fn(outputs, labels)
        
        # Backward propagation
        loss.backward()
        
        # Gradient clipping
        if config.GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, loss_fn, device, config, epoch):
    """Validate"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]')
        
        for frames, arm_images_list, labels in pbar:
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Move each tensor in arm_images_list to device
            arm_images_list_device = []
            for arm_images in arm_images_list:
                arm_images_device = [img.to(device) for img in arm_images]
                arm_images_list_device.append(arm_images_device)
            
            outputs = model(frames, arm_images_list_device)
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            current_loss = running_loss / len(dataloader)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def save_checkpoint(model, optimizer, scheduler, epoch, loss, acc, is_best, config):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'config': {
            'num_classes': config.NUM_CLASSES,
            'num_frames': config.NUM_FRAMES,
            'frame_size': (384, 384),
            'arm_size': (192, 192),
            'model_name': 'DualNetwork',
            'fusion_method': config.FUSION_METHOD,
            'r3d_weight': config.R3D_WEIGHT,
            'arm_weight': config.ARM_WEIGHT
        }
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save latest model
    torch.save(checkpoint, config.LAST_MODEL_PATH)
    
    # Save best model
    if is_best:
        torch.save(checkpoint, config.BEST_MODEL_PATH)
        print(f"Saved best model (accuracy: {acc:.2f}%)")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint.get('accuracy', 0.0)
    
    return start_epoch, best_acc


def main():
    """Main function"""
    # Set random seed
    utils.set_seed(Config.SEED)
    
    # Create directories
    Config.create_dirs()
    
    # Set device
    if Config.DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{Config.CUDA_DEVICE}')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    
    # Preprocessed data directory
    PREPROCESSED_DIR = Path('preprocessed_frames')
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = create_dataloader(
        csv_file=str(Config.TRAIN_CSV),
        preprocessed_dir=str(PREPROCESSED_DIR),
        batch_size=Config.BATCH_SIZE,
        num_frames=Config.NUM_FRAMES,
        frame_size=(384, 384),
        arm_size=(192, 192),
        is_training=True,
        num_workers=Config.NUM_WORKERS,
        class_ind_file=str(Config.CLASS_IND_FILE)
    )
    
    val_loader = create_dataloader(
        csv_file=str(Config.VAL_CSV),
        preprocessed_dir=str(PREPROCESSED_DIR),
        batch_size=Config.BATCH_SIZE,
        num_frames=Config.NUM_FRAMES,
        frame_size=(384, 384),
        arm_size=(192, 192),
        is_training=False,
        num_workers=Config.NUM_WORKERS,
        class_ind_file=str(Config.CLASS_IND_FILE)
    )
    
    # Create model
    print("\nCreating fusion model...")
    model = get_dual_model(
        num_classes=Config.NUM_CLASSES,
        r3d_pretrained=Config.PRETRAINED,
        fusion_method=Config.FUSION_METHOD,
        r3d_weight=Config.R3D_WEIGHT,
        arm_weight=Config.ARM_WEIGHT
    )
    model = model.to(device)
    
    # Print model information
    model_info = model.get_model_info()
    print(f"Model: {model_info['model_name']}")
    print(f"Fusion method: {model_info['fusion_method']}")
    print(f"Fusion weights: R3D={Config.R3D_WEIGHT}, ArmCNN={Config.ARM_WEIGHT}")
    print(f"Number of classes: {model_info['num_classes']}")
    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Trainable parameters: {model_info['trainable_params']:,}")
    print(f"R3D parameters: {model_info['r3d_params']:,}")
    print(f"Arm CNN parameters: {model_info['arm_cnn_params']:,}")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, Config)
    scheduler = get_scheduler(optimizer, Config, len(train_loader))
    
    # Create loss function
    loss_fn = get_loss_function(Config)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Check for checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    
    if Config.LAST_MODEL_PATH.exists():
        print(f"\nFound checkpoint: {Config.LAST_MODEL_PATH}")
        response = input("Continue training from checkpoint? (y/n): ")
        if response.lower() == 'y':
            start_epoch, best_val_acc = load_checkpoint(model, optimizer, scheduler, Config.LAST_MODEL_PATH, device)
            print(f"Resume training from epoch {start_epoch}, best validation accuracy: {best_val_acc:.2f}%")
    
    # Early stopping
    patience_counter = 0
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device, Config, epoch)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, loss_fn, device, Config, epoch)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc, is_best, Config)
        
        # Print information
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        if scheduler is not None:
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nValidation accuracy did not improve for {Config.EARLY_STOPPING_PATIENCE} epochs, stopping early")
            break
    
    # Save training history
    history_path = Config.LOG_DIR / 'training_history_dual.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    # Training time
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()

