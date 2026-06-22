"""
Dataset splitting script
Split training_set into train/validation/test sets in 80/10/10 ratio
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_labels(csv_path):
    """Load label file"""
    df = pd.read_csv(csv_path, header=None, names=['video_id', 'class_name', 'class_id'])
    return df

def split_dataset(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split dataset, ensure balanced class distribution and each set contains all classes
    
    Args:
        df: DataFrame containing video IDs and labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
    
    Returns:
        train_df, val_df, test_df
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Set random seed
    np.random.seed(random_state)
    
    # Group by class, ensure each class is split proportionally and each set contains all classes
    train_list = []
    val_list = []
    test_list = []
    
    for class_id in df['class_id'].unique():
        class_data = df[df['class_id'] == class_id].copy()
        class_data = class_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n_samples = len(class_data)
        
        # Ensure each set contains at least 1 sample (if n_samples >= 3)
        if n_samples >= 3:
            # Allocate at least 1 sample to each set
            min_per_split = 1
            remaining_samples = n_samples - 3  # Subtract the minimum 1 for each set
            
            # Calculate number of samples each set should get (based on remaining samples proportionally)
            train_count = min_per_split + int(remaining_samples * train_ratio)
            val_count = min_per_split + int(remaining_samples * val_ratio)
            test_count = min_per_split + int(remaining_samples * test_ratio)
            
            # Handle remainder to ensure all samples are allocated
            total_allocated = train_count + val_count + test_count
            remainder = n_samples - total_allocated
            
            # Allocate remainder to training set (training set usually needs more samples)
            if remainder > 0:
                train_count += remainder
            
        elif n_samples == 2:
            # 2 samples: 1 for training, 1 for validation, 0 for test
            # Allocate this way first, will adjust from training set later
            train_count = 1
            val_count = 1
            test_count = 0
        else:  # n_samples == 1
            # 1 sample: can only assign to training set, will adjust from other classes later
            train_count = 1
            val_count = 0
            test_count = 0
        
        # Split data
        train_data = class_data.iloc[:train_count].copy()
        val_data = class_data.iloc[train_count:train_count + val_count].copy() if val_count > 0 else pd.DataFrame(columns=class_data.columns)
        test_data = class_data.iloc[train_count + val_count:].copy() if test_count > 0 else pd.DataFrame(columns=class_data.columns)
        
        train_list.append(train_data)
        if val_count > 0:
            val_list.append(val_data)
        if test_count > 0:
            test_list.append(test_data)
    
    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=df.columns)
    
    # Check and fix: ensure each set contains all classes
    all_class_ids = set(df['class_id'].unique())
    val_class_ids = set(val_df['class_id'].unique()) if len(val_df) > 0 else set()
    test_class_ids = set(test_df['class_id'].unique()) if len(test_df) > 0 else set()
    
    missing_in_val = all_class_ids - val_class_ids
    missing_in_test = all_class_ids - test_class_ids
    
    # Borrow samples from training set to validation set (ensure validation set contains all classes)
    for class_id in missing_in_val:
        train_class_samples = train_df[train_df['class_id'] == class_id]
        if len(train_class_samples) > 0:
            # Take 1 sample from training set for validation set
            sample_to_move = train_class_samples.iloc[0:1]
            val_df = pd.concat([val_df, sample_to_move], ignore_index=True)
            train_df = train_df.drop(train_class_samples.index[0])
    
    # Borrow samples from training set to test set (ensure test set contains all classes)
    for class_id in missing_in_test:
        train_class_samples = train_df[train_df['class_id'] == class_id]
        if len(train_class_samples) > 0:
            # Take 1 sample from training set for test set
            sample_to_move = train_class_samples.iloc[0:1]
            test_df = pd.concat([test_df, sample_to_move], ignore_index=True)
            train_df = train_df.drop(train_class_samples.index[0])
    
    # Shuffle order
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df, output_dir='.'):
    """Save split results to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, header=False)
    
    all_df = pd.concat([train_df, val_df, test_df])
    total_samples = len(all_df)
    total_classes = all_df['class_id'].nunique()
    
    print(f"Dataset splitting complete!")
    print(f"Training set: {len(train_df)} samples ({len(train_df)/total_samples*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/total_samples*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/total_samples*100:.1f}%)")
    
    # Verify each set contains all classes
    train_classes = train_df['class_id'].nunique()
    val_classes = val_df['class_id'].nunique()
    test_classes = test_df['class_id'].nunique()
    
    print(f"\nClass coverage:")
    print(f"Total classes: {total_classes}")
    print(f"Training set classes: {train_classes} {'✓' if train_classes == total_classes else '✗'}")
    print(f"Validation set classes: {val_classes} {'✓' if val_classes == total_classes else '✗'}")
    print(f"Test set classes: {test_classes} {'✓' if test_classes == total_classes else '✗'}")
    
    # Print class distribution
    print("\nClass distribution:")
    print("Class ID | Training | Validation | Test | Total")
    print("-" * 50)
    for class_id in sorted(all_df['class_id'].unique()):
        train_count = len(train_df[train_df['class_id'] == class_id])
        val_count = len(val_df[val_df['class_id'] == class_id])
        test_count = len(test_df[test_df['class_id'] == class_id])
        total_count = train_count + val_count + test_count
        # Check if each set has this class
        status = "✓" if (train_count > 0 and val_count > 0 and test_count > 0) else "✗"
        print(f"{class_id:6d} | {train_count:6d} | {val_count:6d} | {test_count:6d} | {total_count:5d} {status}")

def main():
    """Main function"""
    # Path configuration
    labels_path = 'annotations/training_set_labels.csv'
    output_dir = 'annotations'
    
    # Load labels
    print("Loading label file...")
    df = load_labels(labels_path)
    print(f"Total {len(df)} video samples")
    print(f"Total {df['class_id'].nunique()} classes")
    
    # Split dataset
    print("\nSplitting dataset...")
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42
    )
    
    # Save results
    print("\nSaving split results...")
    save_splits(train_df, val_df, test_df, output_dir)
    
    print(f"\nSplit results saved to {output_dir} directory")

if __name__ == '__main__':
    main()

