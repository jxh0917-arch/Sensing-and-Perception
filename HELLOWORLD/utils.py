"""
Utility functions module
Contains video processing, evaluation metrics and visualization functions
"""

import numpy as np
import torch
import cv2
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import os

def sample_frames(video_path: str, num_frames: int, mode: str = 'uniform') -> List[np.ndarray]:
    """
    Sample frames from video
    
    Args:
        video_path: Video file path
        num_frames: Number of frames to sample
        mode: Sampling mode ('uniform' or 'random')
    
    Returns:
        Frame list, each frame is a numpy array (H, W, C)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video file is empty: {video_path}")
    
    # Determine sampling indices
    if mode == 'uniform':
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif mode == 'random':
        indices = np.sort(np.random.choice(total_frames, min(num_frames, total_frames), replace=False))
    else:
        raise ValueError(f"Unsupported sampling mode: {mode}")
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # If read fails, use last frame
            if frames:
                frames.append(frames[-1])
            else:
                # If no frames, create a black frame
                h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    
    cap.release()
    
    # If not enough frames, repeat last frame
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((112, 112, 3), dtype=np.uint8))
    
    return frames[:num_frames]

def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize frame"""
    return cv2.resize(frame, size)

def center_crop(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Center crop"""
    h, w = frame.shape[:2]
    th, tw = size
    
    if h < th or w < tw:
        # If frame is too small, enlarge first
        scale = max(th / h, tw / w)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        h, w = frame.shape[:2]
    
    x1 = (w - tw) // 2
    y1 = (h - th) // 2
    return frame[y1:y1+th, x1:x1+tw]

def random_crop(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Random crop"""
    h, w = frame.shape[:2]
    th, tw = size
    
    if h < th or w < tw:
        # If frame is too small, enlarge first
        scale = max(th / h, tw / w)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        h, w = frame.shape[:2]
    
    x1 = np.random.randint(0, max(1, w - tw))
    y1 = np.random.randint(0, max(1, h - th))
    return frame[y1:y1+th, x1:x1+tw]

def horizontal_flip(frame: np.ndarray) -> np.ndarray:
    """Horizontal flip"""
    return cv2.flip(frame, 1)

def normalize_frames(frames: List[np.ndarray], mean: Tuple[float, float, float] = (0.43216, 0.394666, 0.37645),
                     std: Tuple[float, float, float] = (0.22803, 0.22145, 0.216989)) -> np.ndarray:
    """
    Normalize frames
    
    Args:
        frames: Frame list
        mean: Mean (R, G, B)
        std: Standard deviation (R, G, B)
    
    Returns:
        Normalized numpy array (T, H, W, C) -> (T, C, H, W)
    """
    frames_array = np.array(frames, dtype=np.float32) / 255.0
    
    # Normalize
    mean = np.array(mean).reshape(1, 1, 1, 3)
    std = np.array(std).reshape(1, 1, 1, 3)
    frames_array = (frames_array - mean) / std
    
    # Convert to (T, C, H, W)
    frames_array = np.transpose(frames_array, (0, 3, 1, 2))
    
    return frames_array

def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    """
    Calculate top-k accuracy
    
    Args:
        outputs: Model output (batch_size, num_classes)
        targets: True labels (batch_size,)
        topk: top-k value tuple
    
    Returns:
        top-k accuracy list
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[dict] = None) -> dict:
    """
    Evaluate prediction results
    
    Args:
        y_true: True labels (should be 0-based indices, i.e., 0 to num_classes-1)
        y_pred: Predicted labels (should be 0-based indices, i.e., 0 to num_classes-1)
        class_names: Class name dictionary (key is original class ID, starting from 1)
    
    Returns:
        Evaluation results dictionary
    """
    # Note: y_true and y_pred are 0-based indices (0 to num_classes-1)
    # But class_names keys are 1-based indices (1 to num_classes)
    # Need to convert 0-based indices to 1-based indices, or use 0-based indices directly
    
    # If class_names provided, need to convert 0-based labels to 1-based labels
    # Or use 0-based indices as labels parameter
    if class_names:
        # Get number of classes
        num_classes = len(class_names)
        # Use 0-based indices as labels parameter (because y_true and y_pred are 0-based)
        label_order = list(range(num_classes))
        # Create 0-based to 1-based mapping for subsequent report generation
        idx_to_class_id = {i: sorted(class_names.keys())[i] for i in range(num_classes)}
    else:
        label_order = None
        idx_to_class_id = None
    
    label_kwargs = {'labels': label_order} if label_order is not None else {}

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
        **label_kwargs,
    )
    
    # Macro and micro averages
    precision_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0, **label_kwargs
    )[0]
    recall_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0, **label_kwargs
    )[1]
    f1_macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0, **label_kwargs
    )[2]
    
    precision_micro = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0, **label_kwargs
    )[0]
    recall_micro = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0, **label_kwargs
    )[1]
    f1_micro = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0, **label_kwargs
    )[2]
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, **label_kwargs)
    
    # Classification report
    if class_names and idx_to_class_id:
        # Generate target_names in 0-based index order, but use 1-based class ID to get class names
        target_names = [class_names.get(idx_to_class_id[i], f'Class {idx_to_class_id[i]}') for i in range(num_classes)]
    else:
        target_names = None
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
        **label_kwargs,
    )
    
    results = {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return results

def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[dict] = None, save_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: Class name dictionary
        save_path: Save path
    """
    plt.figure(figsize=(20, 16))
    
    if class_names:
        # Confusion matrix is arranged in 0-based index order
        # Need to generate labels in 0-based index order, but use 1-based class ID to get class names
        sorted_class_ids = sorted(class_names.keys())
        labels = [class_names.get(sorted_class_ids[i], f'Class {sorted_class_ids[i]}') for i in range(len(cm))]
    else:
        labels = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_evaluation_report(results: dict, save_path: str, class_names: Optional[dict] = None):
    """
    Save evaluation report
    
    Args:
        results: Evaluation results dictionary
        save_path: Save path
        class_names: Class name dictionary
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n\n")
        
        f.write("Macro Average Metrics:\n")
        f.write(f"  Precision: {results['precision_macro']:.4f}\n")
        f.write(f"  Recall: {results['recall_macro']:.4f}\n")
        f.write(f"  F1 Score: {results['f1_macro']:.4f}\n\n")
        
        f.write("Micro Average Metrics:\n")
        f.write(f"  Precision: {results['precision_micro']:.4f}\n")
        f.write(f"  Recall: {results['recall_micro']:.4f}\n")
        f.write(f"  F1 Score: {results['f1_micro']:.4f}\n\n")
        
        if class_names:
            f.write("Per-Class Detailed Metrics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Class':<30} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Support':<10}\n")
            f.write("-" * 80 + "\n")
            
            # Create mapping from 0-based index to 1-based class ID
            sorted_class_ids = sorted(class_names.keys())
            for i in range(len(sorted_class_ids)):
                class_id = sorted_class_ids[i]  # 1-based class ID
                class_name = class_names[class_id]
                precision = results['precision_per_class'][i]
                recall = results['recall_per_class'][i]
                f1 = results['f1_per_class'][i]
                support = results['support_per_class'][i]
                f.write(f"{class_name:<30} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Evaluation report saved to: {save_path}")

def set_seed(seed: int = 42):
    """Set random seed"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

