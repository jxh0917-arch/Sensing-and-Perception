"""
Test script
Evaluate model on test set and generate test report
Using dual model (R3D-18 + CNN)
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

from config import Config
from dual_model import load_dual_model
from preprocessed_dataset import create_dataloader
import utils

def test_model(model_path: str = None, save_results: bool = True):
    """
    Test model on test set
    
    Args:
        model_path: Model path (if None, use best model)
        save_results: Whether to save results
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
    model = load_dual_model(model_path, device=str(device))
    print("Model loaded")
    
    # Preprocessed data directory
    PREPROCESSED_DIR = Path('preprocessed_frames')
    
    # Create data loader
    print("\nCreating test data loader...")
    test_loader = create_dataloader(
        csv_file=str(Config.TEST_CSV),
        preprocessed_dir=str(PREPROCESSED_DIR),
        batch_size=Config.BATCH_SIZE,
        num_frames=Config.NUM_FRAMES,
        frame_size=(384, 384),
        arm_size=(192, 192),
        is_training=False,
        num_workers=Config.NUM_WORKERS,
        class_ind_file=str(Config.CLASS_IND_FILE),
        shuffle=False
    )
    
    # Get class names
    class_names = Config.get_class_names()
    
    # Test
    print("\nStarting test...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for frames, arm_images_list, labels in tqdm(test_loader, desc='Testing'):
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Move each tensor in arm_images_list to device
            arm_images_list_device = []
            for arm_images in arm_images_list:
                arm_images_device = [img.to(device) for img in arm_images]
                arm_images_list_device.append(arm_images_device)
            
            outputs = model(frames, arm_images_list_device)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate evaluation metrics
    print("\nCalculating evaluation metrics...")
    results = utils.evaluate_predictions(all_labels, all_preds, class_names)
    
    # Print results
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)
    print(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"\nMacro Average Metrics:")
    print(f"  Precision: {results['precision_macro']:.4f}")
    print(f"  Recall: {results['recall_macro']:.4f}")
    print(f"  F1 Score: {results['f1_macro']:.4f}")
    print(f"\nMicro Average Metrics:")
    print(f"  Precision: {results['precision_micro']:.4f}")
    print(f"  Recall: {results['recall_micro']:.4f}")
    print(f"  F1 Score: {results['f1_micro']:.4f}")
    
    # Print per-class detailed metrics
    if class_names:
        print("\nPer-Class Detailed Metrics:")
        print("-" * 80)
        print(f"{'Class':<30} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Support':<10}")
        print("-" * 80)
        
        # Create mapping from 0-based index to 1-based class ID
        sorted_class_ids = sorted(class_names.keys())
        for i in range(len(sorted_class_ids)):
            class_id = sorted_class_ids[i]  # 1-based class ID
            class_name = class_names[class_id]
            precision = results['precision_per_class'][i]
            recall = results['recall_per_class'][i]
            f1 = results['f1_per_class'][i]
            support = results['support_per_class'][i]
            print(f"{class_name:<30} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
    
    # Save results
    if save_results:
        Config.RESULTS_DIR.mkdir(exist_ok=True)
        
        # Save evaluation report
        report_path = Config.RESULTS_DIR / 'test_report.txt'
        utils.save_evaluation_report(results, str(report_path), class_names)
        
        # Save confusion matrix
        cm_path = Config.RESULTS_DIR / 'test_confusion_matrix.png'
        utils.plot_confusion_matrix(results['confusion_matrix'], class_names, str(cm_path))
        
        # Save prediction results
        results_path = Config.RESULTS_DIR / 'test_predictions.npz'
        np.savez(
            results_path,
            predictions=all_preds,
            labels=all_labels,
            probabilities=all_probs
        )
        print(f"\nPrediction results saved to: {results_path}")
        
        # Save classification report (CSV format)
        csv_path = Config.RESULTS_DIR / 'test_classification_report.csv'
        import pandas as pd
        report_data = []
        # Create mapping from 0-based index to 1-based class ID
        sorted_class_ids = sorted(class_names.keys())
        for i in range(len(sorted_class_ids)):
            class_id = sorted_class_ids[i]  # 1-based class ID
            class_name = class_names[class_id]
            report_data.append({
                'Class ID': class_id,
                'Class Name': class_name,
                'Precision': results['precision_per_class'][i],
                'Recall': results['recall_per_class'][i],
                'F1-Score': results['f1_per_class'][i],
                'Support': results['support_per_class'][i]
            })
        df = pd.DataFrame(report_data)
        df.to_csv(csv_path, index=False)
        print(f"Classification report (CSV) saved to: {csv_path}")
        
        print(f"\nAll results saved to: {Config.RESULTS_DIR}")
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--model', type=str, default=None, help='Model path (default: use best model)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    test_model(model_path=args.model, save_results=not args.no_save)

if __name__ == '__main__':
    main()

