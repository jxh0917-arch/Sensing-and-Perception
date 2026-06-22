"""
Test set application script
Classify videos in test_set directory using trained model
Using dual model (R3D-18 + CNN)
"""

import os
import re
import torch
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from config import Config
from dual_model import load_dual_model
import utils

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


def classify_video(model, video_id: str, preprocessed_dir: Path, device: torch.device) -> tuple:
    """
    Classify a single video (load from preprocessed frame directory)
    
    Args:
        model: Trained model
        video_id: Video ID (folder name)
        preprocessed_dir: Preprocessed frame folder directory
        device: Computing device
    
    Returns:
        (predicted_class_id, predicted_idx, probability): Predicted class ID, index and probability
    """
    # Build video folder path
    video_dir = preprocessed_dir / video_id
    
    if not video_dir.exists():
        raise FileNotFoundError(f"Video folder does not exist: {video_dir}")
    
    # Data transformation (test mode, no data augmentation)
    frame_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                           std=[0.22803, 0.22145, 0.216989])
    ])
    
    arm_transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load 16 action frames
    frames = []
    num_frames = Config.NUM_FRAMES
    for i in range(num_frames):
        # Find corresponding frame's person image
        frame_files = sorted(video_dir.glob(f"frame_{i:03d}_person_*.png"))
        if frame_files:
            # If multiple persons, take the first one
            frame_path = frame_files[0]
            img = Image.open(frame_path).convert('RGB')
            img_tensor = frame_transform(img)
            frames.append(img_tensor)
        else:
            # If not found, use black placeholder
            placeholder = Image.new('RGB', (384, 384), (0, 0, 0))
            img_tensor = frame_transform(placeholder)
            frames.append(img_tensor)
    
    # Convert to (C, T, H, W)
    frames_tensor = torch.stack(frames, dim=1).unsqueeze(0).to(device)  # (1, C, T, H, W)
    
    # Load arm images (ignore pure black placeholder images)
    arm_images = []
    arm_files = sorted(video_dir.glob("frame_*_arm_*.png"))
    for arm_file in arm_files:
        # Check if it's a black image
        if not is_black_image(str(arm_file)):
            img = Image.open(arm_file).convert('RGB')
            img_tensor = arm_transform(img)
            arm_images.append(img_tensor)
    
    # If no valid arm images, create a placeholder (all-zero tensor)
    if len(arm_images) == 0:
        placeholder = Image.new('RGB', (192, 192), (0, 0, 0))
        arm_images.append(arm_transform(placeholder))
    
    # Wrap arm image list in batch format (only one sample) and move to device
    arm_images_device = [img.to(device) for img in arm_images]
    arm_images_list = [arm_images_device]
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(frames_tensor, arm_images_list)
        probs = torch.softmax(outputs, dim=1)
        probability, predicted_idx = probs.max(1)
    
    # Convert to numpy
    predicted_idx = predicted_idx.cpu().item()  # 0-based index
    probability = probability.cpu().item()
    
    # Convert to 1-based class ID (because IDs in classInd.txt start from 1)
    predicted_class_id = predicted_idx + 1
    
    return predicted_class_id, predicted_idx, probability

def classify_test_set(model_path: str = None, output_csv: str = None, batch_size: int = None):
    """
    Classify all videos in test_set directory
    
    Args:
        model_path: Model path (if None, use best model)
        output_csv: Output CSV file path (if None, use default path)
        batch_size: Batch size (optional, for optimizing processing speed)
    """
    # Set device
    if Config.DEVICE == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{Config.CUDA_DEVICE}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    if model_path is None:
        model_path = Config.BEST_MODEL_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    print(f"\nLoading model: {model_path}")
    model = load_dual_model(
        model_path,
        device=str(device),
        fusion_method=Config.FUSION_METHOD,
        r3d_weight=Config.R3D_WEIGHT,
        arm_weight=Config.ARM_WEIGHT
    )
    print("Model loaded")
    
    # Load class name mapping
    class_names = Config.get_class_names()
    if not class_names:
        raise ValueError("Cannot load class names, please check classInd.txt file")
    
    print(f"\nNumber of classes: {len(class_names)}")
    
    # Preprocessed test frame directory
    preprocessed_test_dir = Path('preprocessed_test_frames')
    if not preprocessed_test_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed test frame directory does not exist: {preprocessed_test_dir}\n"
            "Please run preprocess_test_video.py to preprocess test videos first"
        )
    
    # Get all preprocessed video folders
    video_dirs = [d for d in preprocessed_test_dir.iterdir() if d.is_dir()]
    if not video_dirs:
        raise ValueError(f"No preprocessed video folders found in {preprocessed_test_dir}")
    
    print(f"\nFound {len(video_dirs)} preprocessed video folders")
    
    # Store results
    results = []
    
    # Classify each video
    print("\nStarting classification...")
    for video_dir in tqdm(video_dirs, desc='Classification progress'):
        try:
            # Get video ID (folder name)
            video_id = video_dir.name
            
            # Classify
            predicted_class_id, predicted_idx, probability = classify_video(
                model, video_id, preprocessed_test_dir, device
            )
            
            # Get class name
            predicted_class_name = class_names.get(predicted_class_id, f"Class_{predicted_class_id}")
            
            # Save result
            results.append({
                'video_id': video_id,
                'class_name': predicted_class_name,
                'class_id': predicted_class_id,
                'probability': probability
            })
            
        except Exception as e:
            print(f"\nWarning: Error processing video {video_id}: {str(e)}")
            # Still record video ID, but mark as error
            results.append({
                'video_id': video_id,
                'class_name': 'ERROR',
                'class_id': -1,
                'probability': 0.0
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by video_id (support formats like test_video001)
    def extract_number(video_id):
        """Extract number from video_id for sorting"""
        # Extract all numbers
        numbers = re.findall(r'\d+', str(video_id))
        if numbers:
            # Return last number (usually video number)
            return int(numbers[-1])
        return 0
    
    try:
        # Try to extract numbers for sorting
        df['video_id_num'] = df['video_id'].apply(extract_number)
        df = df.sort_values('video_id_num')
        df = df.drop(columns=['video_id_num'])
    except:
        # If fails, sort by string
        df = df.sort_values('video_id')
    
    # Save results
    if output_csv is None:
        output_csv = Config.RESULTS_DIR / 'test_set_labels.csv'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save as CSV, format consistent with CSV in annotations: video_id,class_name,class_id (no header)
    output_df = df[['video_id', 'class_name', 'class_id']].copy()
    output_df.to_csv(output_csv, index=False, header=False)
    
    # Also save a detailed version with probabilities (with header)
    detailed_csv = str(output_csv).replace('.csv', '_detailed.csv')
    df.to_csv(detailed_csv, index=False)
    
    print(f"\nClassification complete!")
    print(f"Results saved to: {output_csv}")
    print(f"Detailed results (with probabilities) saved to: {detailed_csv}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Classification Statistics")
    print("=" * 80)
    print(f"Total videos: {len(df)}")
    
    # Count number of each class
    class_counts = df[df['class_id'] > 0]['class_name'].value_counts().sort_index()
    print(f"\nClassification count by class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Count errors
    error_count = len(df[df['class_id'] == -1])
    if error_count > 0:
        print(f"\nFailed videos: {error_count}")
    
    # Statistics on average confidence
    valid_probs = df[df['probability'] > 0]['probability']
    if len(valid_probs) > 0:
        print(f"\nAverage confidence: {valid_probs.mean():.4f}")
        print(f"Minimum confidence: {valid_probs.min():.4f}")
        print(f"Maximum confidence: {valid_probs.max():.4f}")
    
    return df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Classify videos in test_set directory')
    parser.add_argument('--model', type=str, default=None, help='Model path (default: use best model)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (not used in current version)')
    
    args = parser.parse_args()
    
    classify_test_set(
        model_path=args.model,
        output_csv=args.output,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()

