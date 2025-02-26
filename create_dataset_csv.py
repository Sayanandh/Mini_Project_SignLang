import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def create_dataset_csv(data_dir, output_dir='.'):
    """
    Create CSV files for train, validation, and test datasets from a directory of images.

    Args:
        data_dir (str): Path to the directory containing images.
        output_dir (str): Path to the directory where CSV files will be saved.
    """
    # Normalize the data directory path
    data_dir = os.path.normpath(data_dir)
    output_dir = os.path.normpath(output_dir)

    # Lists to store filenames and labels
    filenames = []
    labels = []

    # Walk through all directories
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Get the label from the directory name
                label = os.path.basename(root)
                # Create relative path that includes the label directory
                rel_path = os.path.join(label, file)
                # Use forward slashes for consistency
                rel_path = rel_path.replace('\\', '/')
                filenames.append(rel_path)
                labels.append(label)

    if not filenames:
        raise ValueError(f"No image files found in {data_dir}")

    # Create DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'label': labels
    })

    # Create one-hot encoded labels
    lb = LabelBinarizer()
    label_classes = sorted(df['label'].unique())  # Sort to ensure consistent ordering
    labels_one_hot = lb.fit_transform(df['label'])

    # Add one-hot encoded columns to DataFrame
    for idx, class_name in enumerate(lb.classes_):
        df[class_name] = labels_one_hot[:, idx]

    print(f"Found {len(df)} images across {len(label_classes)} classes")
    print(f"Classes: {', '.join(label_classes)}")

    # Split into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_dir, 'augmented'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    # Save CSVs in their respective directories
    train_csv_path = os.path.join(output_dir, 'augmented', 'train_labels.csv')
    val_csv_path = os.path.join(output_dir, 'valid', 'val_labels.csv')
    test_csv_path = os.path.join(output_dir, 'test', 'test_labels.csv')

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"\nData split complete:")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Testing images: {len(test_df)}")
    print(f"\nCSV files have been saved in their respective directories:")
    print(f"Train: {train_csv_path}")
    print(f"Validation: {val_csv_path}")
    print(f"Test: {test_csv_path}")
    print(f"\nColumns in CSV files:")
    print(f"- filename: Image path")
    print(f"- label: Original label")
    print(f"- One-hot encoded columns: {', '.join(lb.classes_)}")

if __name__ == "__main__":
    # Use relative path to the data directory containing all images
    data_dir = "A:/Signlang_gpy/augmented"  # Change this to your image directory path
    output_dir = "A:/Signlang_gpy"  # Change this to your desired output directory
    create_dataset_csv(data_dir, output_dir)