import os
import shutil
import random
from pathlib import Path

# ==== CONFIGURATION ====
SOURCE_IMAGES_DIR = "E:/dataset_img/output"
SOURCE_ANNOTATIONS_DIR = "E:/dataset_img/annotations"
OUTPUT_BASE_DIR = "E:/dataset_img/yolo_dataset"

# Split ratios
TRAIN_RATIO = 0.82
VAL_RATIO = 0.15
TEST_RATIO = 0.03

# Optional: Reduce dataset size (set to None to use full dataset)
MAX_IMAGES = 25000  # Use only 25k images instead of 60k (set to None for full dataset)

# Random seed for reproducible splits
RANDOM_SEED = 42


def create_directory_structure(base_dir):
    """Create YOLOv8 standard directory structure"""
    dirs_to_create = [
        "train/images",
        "train/labels",
        "val/images",
        "val/labels",
        "test/images",
        "test/labels"
    ]

    for dir_path in dirs_to_create:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")


def get_image_annotation_pairs(images_dir, annotations_dir):
    """Get list of (image_path, annotation_path) pairs that exist in both directories"""
    image_files = set()
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.update(Path(images_dir).glob(ext))

    # Get basenames without extension
    image_basenames = {f.stem for f in image_files}

    # Find corresponding annotation files
    pairs = []
    for basename in image_basenames:
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = Path(images_dir) / f"{basename}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break

        ann_path = Path(annotations_dir) / f"{basename}.txt"

        if img_path and ann_path.exists():
            pairs.append((str(img_path), str(ann_path)))
        else:
            print(f"Warning: Missing pair for {basename}")

    return pairs


def balanced_split_by_terrain_and_mines(pairs, train_ratio, val_ratio, test_ratio):
    """
    Create balanced splits ensuring each terrain and mine combination
    appears proportionally in train/val/test sets
    """
    # Group pairs by terrain type and mine configuration
    terrain_mine_groups = {}

    for img_path, ann_path in pairs:
        filename = Path(img_path).stem

        # Extract terrain and mine info from filename
        # Format: terrainname_mine0_single0 or terrainname_double_0_single0
        parts = filename.split('_')
        terrain_name = parts[0]

        if 'double' in filename:
            mine_config = 'double'
        elif 'triple' in filename:
            mine_config = 'triple'
        elif 'quad' in filename:
            mine_config = 'quad'
        else:
            mine_config = 'single'

        group_key = f"{terrain_name}_{mine_config}"

        if group_key not in terrain_mine_groups:
            terrain_mine_groups[group_key] = []
        terrain_mine_groups[group_key].append((img_path, ann_path))

    print(f"Found {len(terrain_mine_groups)} terrain-mine groups")

    # Split each group proportionally
    train_pairs = []
    val_pairs = []
    test_pairs = []

    for group_key, group_pairs in terrain_mine_groups.items():
        random.shuffle(group_pairs)
        n = len(group_pairs)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        group_train = group_pairs[:train_end]
        group_val = group_pairs[train_end:val_end]
        group_test = group_pairs[val_end:]

        train_pairs.extend(group_train)
        val_pairs.extend(group_val)
        test_pairs.extend(group_test)

        print(f"Group {group_key}: {len(group_train)} train, {len(group_val)} val, {len(group_test)} test")

    return train_pairs, val_pairs, test_pairs


def copy_files(pairs, dest_images_dir, dest_labels_dir):
    """Copy image and annotation files to destination directories"""
    success_count = 0
    for img_path, ann_path in pairs:
        try:
            # Copy image
            img_filename = Path(img_path).name
            shutil.copy2(img_path, os.path.join(dest_images_dir, img_filename))

            # Copy annotation
            ann_filename = Path(ann_path).name
            shutil.copy2(ann_path, os.path.join(dest_labels_dir, ann_filename))

            success_count += 1
        except Exception as e:
            print(f"Error copying {img_path}: {e}")

    return success_count


def create_dataset_yaml(base_dir, train_count, val_count, test_count):
    """Create dataset.yaml file for YOLOv8"""
    yaml_content = f"""# YOLOv8 Landmine Detection Dataset
# Dataset created from synthetic data generation

# Paths (relative to this file)
path: {base_dir}
train: train/images
val: val/images  
test: test/images

# Classes
nc: 1  # number of classes
names: ['landmine']  # class names

# Dataset statistics
train_images: {train_count}
val_images: {val_count}
test_images: {test_count}
total_images: {train_count + val_count + test_count}

# Dataset info
description: "Synthetic landmine detection dataset generated from {45} terrain images and {30} landmine images"
version: "1.0"
"""

    yaml_path = os.path.join(base_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created dataset.yaml at: {yaml_path}")


def main():
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)

    print("YOLOv8 Dataset Split Generator")
    print("=" * 50)

    # Check source directories exist
    if not os.path.exists(SOURCE_IMAGES_DIR):
        print(f"Error: Source images directory not found: {SOURCE_IMAGES_DIR}")
        return

    if not os.path.exists(SOURCE_ANNOTATIONS_DIR):
        print(f"Error: Source annotations directory not found: {SOURCE_ANNOTATIONS_DIR}")
        return

    # Create output directory structure
    print(f"Creating dataset structure in: {OUTPUT_BASE_DIR}")
    create_directory_structure(OUTPUT_BASE_DIR)

    # Get image-annotation pairs
    print("\nFinding image-annotation pairs...")
    all_pairs = get_image_annotation_pairs(SOURCE_IMAGES_DIR, SOURCE_ANNOTATIONS_DIR)
    print(f"Found {len(all_pairs)} valid image-annotation pairs")

    # Optionally reduce dataset size
    if MAX_IMAGES and len(all_pairs) > MAX_IMAGES:
        print(f"Reducing dataset from {len(all_pairs)} to {MAX_IMAGES} images")
        random.shuffle(all_pairs)
        all_pairs = all_pairs[:MAX_IMAGES]

    # Perform balanced split
    print("\nPerforming balanced dataset split...")
    train_pairs, val_pairs, test_pairs = balanced_split_by_terrain_and_mines(
        all_pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    print(f"\nSplit summary:")
    print(f"Train: {len(train_pairs)} images ({len(train_pairs) / len(all_pairs) * 100:.1f}%)")
    print(f"Val:   {len(val_pairs)} images ({len(val_pairs) / len(all_pairs) * 100:.1f}%)")
    print(f"Test:  {len(test_pairs)} images ({len(test_pairs) / len(all_pairs) * 100:.1f}%)")
    print(f"Total: {len(all_pairs)} images")

    # Copy files to destination
    print("\nCopying files...")

    # Train set
    train_img_dir = os.path.join(OUTPUT_BASE_DIR, "train", "images")
    train_lbl_dir = os.path.join(OUTPUT_BASE_DIR, "train", "labels")
    train_copied = copy_files(train_pairs, train_img_dir, train_lbl_dir)
    print(f"Copied {train_copied} training pairs")

    # Validation set
    val_img_dir = os.path.join(OUTPUT_BASE_DIR, "val", "images")
    val_lbl_dir = os.path.join(OUTPUT_BASE_DIR, "val", "labels")
    val_copied = copy_files(val_pairs, val_img_dir, val_lbl_dir)
    print(f"Copied {val_copied} validation pairs")

    # Test set
    test_img_dir = os.path.join(OUTPUT_BASE_DIR, "test", "images")
    test_lbl_dir = os.path.join(OUTPUT_BASE_DIR, "test", "labels")
    test_copied = copy_files(test_pairs, test_img_dir, test_lbl_dir)
    print(f"Copied {test_copied} test pairs")

    # Create dataset.yaml
    create_dataset_yaml(OUTPUT_BASE_DIR, train_copied, val_copied, test_copied)

    print("\n" + "=" * 50)
    print("Dataset split completed successfully!")
    print(f"Dataset location: {OUTPUT_BASE_DIR}")
    print(f"Use dataset.yaml for YOLOv8 training")

    # Print training command
    print(f"\nTo train YOLOv8 nano:")
    print(f"yolo train model=yolov8n.pt data='{OUTPUT_BASE_DIR}/dataset.yaml' epochs=100 imgsz=640")


if __name__ == "__main__":
    main()