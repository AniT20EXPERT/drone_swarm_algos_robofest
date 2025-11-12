import cv2
import numpy as np
import os
import random
import json
from PIL import Image, ImageEnhance, ImageFilter

# ==== CONFIG ====
TERRAIN_DIR = "./terrains"  # folder with terrain JPG/PNG
MINE_DIR = "./landmines"  # folder with landmine PNG (with alpha)
OUTPUT_DIR = "E:/dataset_img/output"
ANNOTATIONS_DIR = "E:/dataset_img/annotations"  # YOLO format annotations
FINAL_SIZE = (640, 640)  # YOLOv8-friendly
MINE_SIZE_FACTOR = 0.5 # make mine size 50 percent of original size
SINGLE_AUG_COUNT = 9  # one augmentation at a time
MULTI_AUG_COUNT = 18  # all augmentations together, random params

# Multiple mines configuration
MIN_MINES_PER_IMAGE = 1
MAX_MINES_PER_IMAGE = 4
MIN_MINE_SEPARATION = 120  # minimum pixels between mine centers (increased)


# ==== FRESH ANNOTATION APPROACH ====
def create_annotations_after_augmentation(original_img, augmented_img, original_bboxes):
    """
    Create new annotations by finding mines in the augmented image
    This is a fresh approach that doesn't rely on mathematical transformations
    """
    # For now, let's use a simple approach:
    # Since we know the original positions and we're applying known transformations,
    # we'll create a mapping function for each transformation type

    # This is a placeholder - we need to implement mine detection or
    # track transformations more carefully
    return original_bboxes  # Temporary fallback


def apply_rotation_to_bbox(bbox, angle, img_w, img_h):
    """Apply rotation to a single bbox - IMPROVED VERSION"""
    class_id, center_x, center_y, width, height = bbox

    # Convert center to pixel coordinates relative to image center
    img_center_x, img_center_y = img_w / 2, img_h / 2

    # Pixel coordinates relative to image center
    px_x = center_x * img_w - img_center_x
    px_y = center_y * img_h - img_center_y

    # Apply rotation (clockwise)
    angle_rad = -np.radians(angle)  # OpenCV uses clockwise, math uses counter-clockwise
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    new_px_x = px_x * cos_a - px_y * sin_a
    new_px_y = px_x * sin_a + px_y * cos_a

    # Convert back to normalized coordinates
    new_center_x = (new_px_x + img_center_x) / img_w
    new_center_y = (new_px_y + img_center_y) / img_h

    # IMPROVED: Adjust bounding box size for rotation to account for orientation changes
    # Calculate the rotated bounding box dimensions
    cos_abs = abs(cos_a)
    sin_abs = abs(sin_a)
    new_width = width * cos_abs + height * sin_abs
    new_height = width * sin_abs + height * cos_abs

    return [class_id, new_center_x, new_center_y, new_width, new_height]


def apply_perspective_to_bbox(bbox, tilt_deg, img_w, img_h):
    """Apply perspective transform to a single bbox - IMPROVED VERSION"""
    class_id, center_x, center_y, width, height = bbox

    # Convert to pixel coordinates
    px_x = center_x * img_w
    px_y = center_y * img_h

    # Use the EXACT same transformation matrix as the image
    src = np.float32([[0, 0], [img_w, 0], [0, img_h], [img_w, img_h]])
    dx = img_w * np.tan(np.radians(tilt_deg)) * 0.1
    dy = img_h * np.tan(np.radians(tilt_deg)) * 0.1
    dst = np.float32([[dx, dy], [img_w - dx, dy], [dx, img_h - dy], [img_w - dx, img_h - dy]])
    M = cv2.getPerspectiveTransform(src, dst)

    # Transform all four corners of the bounding box
    half_w = width * img_w / 2
    half_h = height * img_h / 2

    corners = np.array([
        [px_x - half_w, px_y - half_h, 1.0],
        [px_x + half_w, px_y - half_h, 1.0],
        [px_x - half_w, px_y + half_h, 1.0],
        [px_x + half_w, px_y + half_h, 1.0]
    ]).T

    # Transform all corners
    transformed_corners = M @ corners
    # Normalize by w coordinate
    for i in range(4):
        transformed_corners[:, i] /= transformed_corners[2, i]

    # Find bounding box of transformed corners
    min_x = np.min(transformed_corners[0, :])
    max_x = np.max(transformed_corners[0, :])
    min_y = np.min(transformed_corners[1, :])
    max_y = np.max(transformed_corners[1, :])

    # Calculate new center and dimensions
    new_center_x = (min_x + max_x) / 2 / img_w
    new_center_y = (min_y + max_y) / 2 / img_h
    new_width = (max_x - min_x) / img_w
    new_height = (max_y - min_y) / img_h

    return [class_id, new_center_x, new_center_y, new_width, new_height]


def apply_scale_to_bbox(bbox, factor, img_w, img_h):
    """Apply scaling to a single bbox - FRESH START"""
    class_id, center_x, center_y, width, height = bbox

    if factor < 1.0:
        # Scaling down: move towards center and shrink
        new_center_x = 0.5 + (center_x - 0.5) * factor
        new_center_y = 0.5 + (center_y - 0.5) * factor
        new_width = width * factor
        new_height = height * factor
    else:
        # Scaling up: center crop, so positions stay the same
        new_center_x = center_x
        new_center_y = center_y
        new_width = width
        new_height = height

    return [class_id, new_center_x, new_center_y, new_width, new_height]


def transform_bbox_list(bboxes, transform_type, param, img_w, img_h):
    """Transform a list of bboxes - IMPROVED VERSION"""
    if not bboxes:
        return []

    transformed = []
    for bbox in bboxes:
        if transform_type == 'rotation':
            new_bbox = apply_rotation_to_bbox(bbox, param, img_w, img_h)
        elif transform_type == 'perspective':
            new_bbox = apply_perspective_to_bbox(bbox, param, img_w, img_h)
        elif transform_type == 'scale':
            new_bbox = apply_scale_to_bbox(bbox, param, img_w, img_h)
        else:
            new_bbox = bbox  # No transformation for other augmentations

        # IMPROVED: More lenient validation - only check if center is within image
        class_id, cx, cy, w, h = new_bbox

        # Clamp bounding box to image boundaries
        cx = max(0.05, min(0.95, cx))  # Keep center away from extreme edges
        cy = max(0.05, min(0.95, cy))  # Keep center away from extreme edges

        # Ensure minimum size
        w = max(0.02, min(0.4, w))  # Minimum 2% of image, maximum 40%
        h = max(0.02, min(0.4, h))  # Minimum 2% of image, maximum 40%

        # Adjust bbox if it extends beyond image boundaries
        half_w = w / 2
        half_h = h / 2

        # Adjust center if bbox extends beyond boundaries
        if cx - half_w < 0:
            cx = half_w
        if cx + half_w > 1:
            cx = 1 - half_w
        if cy - half_h < 0:
            cy = half_h
        if cy + half_h > 1:
            cy = 1 - half_h

        # Final validation - only reject if bbox is completely invalid
        if 0 <= cx <= 1 and 0 <= cy <= 1 and w > 0 and h > 0:
            transformed.append([class_id, cx, cy, w, h])

    return transformed


def change_lighting(img, brightness_factor, contrast_factor):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness_factor)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_factor)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def motion_blur(img, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel)


def gaussian_blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def gaussian_noise(img, var):
    row, col, ch = img.shape
    mean = 0
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(img + gauss * 255, 0, 255).astype(np.uint8)
    return noisy


def salt_pepper_noise(img, amount):
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = [255, 255, 255]
    num_pepper = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = [0, 0, 0]
    return noisy


def perspective_transform(img, tilt_deg, original_terrain=None):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dx = w * np.tan(np.radians(tilt_deg)) * 0.1
    dy = h * np.tan(np.radians(tilt_deg)) * 0.1
    dst = np.float32([[dx, dy], [w - dx, dy], [dx, h - dy], [w - dx, h - dy]])
    M = cv2.getPerspectiveTransform(src, dst)
    transformed = cv2.warpPerspective(img, M, (w, h))

    if original_terrain is not None:
        # Fill black areas using ONLY the original terrain (no mines)
        mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, M, (w, h))
        mask_inv = cv2.bitwise_not(mask)

        # Scale original terrain and blur for background
        background = cv2.resize(original_terrain, (int(w * 1.2), int(h * 1.2)))
        start_x = (background.shape[1] - w) // 2
        start_y = (background.shape[0] - h) // 2
        background = background[start_y:start_y + h, start_x:start_x + w]
        background = cv2.GaussianBlur(background, (15, 15), 0)

        # Combine transformed image with terrain background
        result = cv2.bitwise_and(transformed, transformed, mask=mask)
        background_part = cv2.bitwise_and(background, background, mask=mask_inv)
        return cv2.add(result, background_part)

    return transformed


def rotate(img, angle, original_terrain=None):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))

    if original_terrain is not None:
        # Create mask to identify black areas
        mask = cv2.warpAffine(np.ones((h, w), dtype=np.uint8) * 255, M, (w, h))
        mask_inv = cv2.bitwise_not(mask)

        # Create background using ONLY original terrain (no mines)
        scale_factor = 1.5  # Scale up to ensure coverage
        background = cv2.resize(original_terrain, (int(w * scale_factor), int(h * scale_factor)))
        start_x = (background.shape[1] - w) // 2
        start_y = (background.shape[0] - h) // 2
        background = background[start_y:start_y + h, start_x:start_x + w]
        background = cv2.GaussianBlur(background, (15, 15), 0)

        # Combine rotated image with terrain background
        result = cv2.bitwise_and(rotated, rotated, mask=mask)
        background_part = cv2.bitwise_and(background, background, mask=mask_inv)
        return cv2.add(result, background_part)

    return rotated


def scale(img, factor, original_terrain=None):
    h, w = img.shape[:2]
    if factor < 1.0:
        # Scaling down - create background first
        scaled = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
        scaled_h, scaled_w = scaled.shape[:2]

        if original_terrain is not None:
            # Create background using ONLY original terrain (no mines)
            background = cv2.resize(original_terrain, (int(w * 1.3), int(h * 1.3)))
            bg_start_x = (background.shape[1] - w) // 2
            bg_start_y = (background.shape[0] - h) // 2
            background = background[bg_start_y:bg_start_y + h, bg_start_x:bg_start_x + w]
            background = cv2.GaussianBlur(background, (21, 21), 0)
        else:
            # Fallback to black background
            background = np.zeros((h, w, 3), dtype=np.uint8)

        # Place scaled image in center of background
        start_x = (w - scaled_w) // 2
        start_y = (h - scaled_h) // 2
        result = background.copy()
        result[start_y:start_y + scaled_h, start_x:start_x + scaled_w] = scaled
        return result
    else:
        # Scaling up - crop from center
        scaled = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
        scaled_h, scaled_w = scaled.shape[:2]
        start_x = (scaled_w - w) // 2
        start_y = (scaled_h - h) // 2
        return scaled[start_y:start_y + h, start_x:start_x + w]


def hue_shift(img, shift_value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift_value) % 180
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def check_overlap(new_pos, existing_positions, min_separation):
    """Check if new mine position overlaps with existing ones"""
    x1, y1, w1, h1 = new_pos
    center_x1 = x1 + w1 / 2
    center_y1 = y1 + h1 / 2

    for pos in existing_positions:
        x2, y2, w2, h2 = pos
        center_x2 = x2 + w2 / 2
        center_y2 = y2 + h2 / 2

        # Calculate distance between centers
        distance = np.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)
        if distance < min_separation:
            return True
    return False


# ==== IMPROVED IMAGE OVERLAY WITH CONFIGURABLE MINE COUNT ====
def overlay_specific_mines(terrain_img, selected_mine_images):
    """
    Overlay specific mine images on terrain and return both image and bounding box annotations
    """
    h, w = terrain_img.shape[:2]
    num_mines = len(selected_mine_images)

    # Convert terrain to PIL for alpha compositing
    terrain_pil = Image.fromarray(cv2.cvtColor(terrain_img, cv2.COLOR_BGR2RGB))

    mine_positions = []  # Store positions for YOLO annotations
    placed_positions = []  # Store positions to avoid overlap

    attempts = 0
    max_attempts = 50  # Prevent infinite loops

    for i, mine_img in enumerate(selected_mine_images):
        if attempts > max_attempts:
            break

        # Random scale for this mine
        base_scale_min = 0.15 if num_mines > 1 else 0.2
        base_scale_max = 0.4 if num_mines > 1 else 0.5
        scale_factor = random.uniform(base_scale_min, base_scale_max) * MINE_SIZE_FACTOR
        mine_resized = mine_img.resize(
            (int(w * scale_factor), int(h * scale_factor)),
            Image.Resampling.LANCZOS
        )

        # Try to find non-overlapping position
        placed = False
        for attempt in range(20):
            max_x = w - mine_resized.width
            max_y = h - mine_resized.height
            if max_x <= 0 or max_y <= 0:
                break

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            new_pos = (x, y, mine_resized.width, mine_resized.height)

            if not check_overlap(new_pos, placed_positions, MIN_MINE_SEPARATION):
                # Place the mine using proper alpha compositing
                terrain_pil.paste(mine_resized, (x, y), mine_resized)

                # Store position for YOLO format (normalized coordinates)
                center_x = (x + mine_resized.width / 2) / w
                center_y = (y + mine_resized.height / 2) / h
                bbox_w = mine_resized.width / w
                bbox_h = mine_resized.height / h

                mine_positions.append([0, center_x, center_y, bbox_w, bbox_h])  # class 0 for landmine
                placed_positions.append(new_pos)
                placed = True
                break

        if not placed:
            attempts += 1

    return cv2.cvtColor(np.array(terrain_pil), cv2.COLOR_RGB2BGR), mine_positions


def generate_mine_combinations(mine_images, num_mines, num_combinations):
    """Generate random combinations of mines"""
    combinations = []
    for _ in range(num_combinations):
        selected_indices = random.sample(range(len(mine_images)), num_mines)
        selected_mines = [mine_images[i] for i in selected_indices]
        combinations.append(selected_mines)
    return combinations


def save_yolo_annotation(annotations, filename):
    """Save YOLO format annotations to text file - IMPROVED with validation"""
    annotation_path = os.path.join(ANNOTATIONS_DIR, f"{filename}.txt")

    # Filter out invalid annotations before saving
    valid_annotations = []
    for ann in annotations:
        class_id, cx, cy, w, h = ann
        # Only save if annotation is valid
        if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
            # Additional check: ensure bbox doesn't extend beyond image
            if (cx - w / 2) >= 0 and (cx + w / 2) <= 1 and (cy - h / 2) >= 0 and (cy + h / 2) <= 1:
                valid_annotations.append(ann)

    with open(annotation_path, 'w') as f:
        for ann in valid_annotations:
            f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

    # Debug: Print annotation count
    if len(valid_annotations) != len(annotations):
        print(f"    Warning: {filename} - Filtered {len(annotations) - len(valid_annotations)} invalid annotations")


# ==== MAIN PIPELINE ====
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(ANNOTATIONS_DIR):
    os.makedirs(ANNOTATIONS_DIR)

terrains = [os.path.join(TERRAIN_DIR, f) for f in os.listdir(TERRAIN_DIR)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
mines = [os.path.join(MINE_DIR, f) for f in os.listdir(MINE_DIR)
         if f.lower().endswith('.png')]

# Load all mine images once
mine_images = []
for mine_path in mines:
    mine_img = Image.open(mine_path).convert("RGBA")
    mine_images.append(mine_img)

print(f"Found {len(terrains)} terrain images and {len(mines)} mine images")
print(f"Generating structured dataset:")
print(f"1. Single mine per image: 20 × 45 × 27 = 24,300 images")
print(f"2. Double mines per image: 10 × 45 × 27 = 12,150 images")
print(f"3. Triple mines per image: 10 × 45 × 27 = 12,150 images")
print(f"4. Quad mines per image: 10 × 45 × 27 = 12,150 images")
print(f"Total: 60,750 images")
print()

image_counter = 0

# ==================== PHASE 1: SINGLE MINE PER IMAGE ====================
print("Phase 1: Generating single mine per image...")

# Select 20 random mines for Phase 1
selected_single_mines = random.sample(mine_images, min(20, len(mine_images)))
print(f"  Selected {len(selected_single_mines)} random mines for single mine images")

for terrain_idx, terrain_path in enumerate(terrains):
    print(f"  Processing terrain {terrain_idx + 1}/{len(terrains)}: {os.path.basename(terrain_path)}")

    terrain_img = cv2.imread(terrain_path)
    terrain_img = cv2.resize(terrain_img, FINAL_SIZE)

    # For each selected mine (REMOVED the variation loop)
    for mine_idx, mine_img in enumerate(selected_single_mines):
        # Create composite with single mine
        composite, mine_annotations = overlay_specific_mines(terrain_img.copy(), [mine_img])

        # Base filename
        base_name = f"{os.path.basename(terrain_path)[:-4]}_mine{mine_idx}"

        # Apply augmentations directly (REMOVED variation loop)
        # SINGLE AUGMENTATIONS
        single_configs = [
            ('lighting', change_lighting, (random.uniform(0.7, 1.3), random.uniform(0.8, 1.2)), None),
            ('motion_blur', motion_blur, (random.randint(3, 7),), None),
            ('gaussian_blur', gaussian_blur, (random.uniform(0, 1.5),), None),
            ('gaussian_noise', gaussian_noise, (random.uniform(0, 0.05),), None),
            ('salt_pepper', salt_pepper_noise, (random.uniform(0, 0.02),), None),
            ('perspective', perspective_transform, (random.uniform(10, 20), terrain_img), 'perspective'),
            ('rotation', rotate, (random.randint(0, 360), terrain_img), 'rotation'),
            ('scale', scale, (random.uniform(0.3, 0.8), terrain_img), 'scale'),
            ('hue_shift', hue_shift, (random.randint(-10, 10),), None),
        ]

        for i, (aug_name, aug_func, params, transform_type) in enumerate(single_configs):
            aug_img = aug_func(composite.copy(), *params)

            if transform_type is None:
                aug_bboxes = mine_annotations.copy()
            else:
                param_value = params[0]
                aug_bboxes = transform_bbox_list(mine_annotations, transform_type, param_value, FINAL_SIZE[0],
                                                 FINAL_SIZE[1])

            filename = f"{base_name}_single{i}"
            cv2.imwrite(f"{OUTPUT_DIR}/{filename}.png", aug_img)
            save_yolo_annotation(aug_bboxes, filename)
            image_counter += 1

        # MULTI AUGMENTATIONS
        for m in range(MULTI_AUG_COUNT):
            img_multi = composite.copy()
            bboxes_multi = mine_annotations.copy()

            tilt_deg = random.uniform(10, 20)
            img_multi = perspective_transform(img_multi, tilt_deg, terrain_img)
            bboxes_multi = transform_bbox_list(bboxes_multi, 'perspective', tilt_deg, FINAL_SIZE[0], FINAL_SIZE[1])

            angle = random.randint(0, 360)
            img_multi = rotate(img_multi, angle, terrain_img)
            bboxes_multi = transform_bbox_list(bboxes_multi, 'rotation', angle, FINAL_SIZE[0], FINAL_SIZE[1])

            scale_factor = random.uniform(0.6, 1.4)
            img_multi = scale(img_multi, scale_factor, terrain_img)
            bboxes_multi = transform_bbox_list(bboxes_multi, 'scale', scale_factor, FINAL_SIZE[0], FINAL_SIZE[1])

            img_multi = change_lighting(img_multi, random.uniform(0.7, 1.3), random.uniform(0.8, 1.2))
            img_multi = motion_blur(img_multi, random.randint(3, 7))
            img_multi = gaussian_blur(img_multi, random.uniform(0, 1.5))
            img_multi = gaussian_noise(img_multi, random.uniform(0, 0.05))
            img_multi = salt_pepper_noise(img_multi, random.uniform(0, 0.02))
            img_multi = hue_shift(img_multi, random.randint(-10, 10))

            filename = f"{base_name}_multi{m}"
            cv2.imwrite(f"{OUTPUT_DIR}/{filename}.png", img_multi)
            save_yolo_annotation(bboxes_multi, filename)
            image_counter += 1

# ==================== PHASE 2-4: MULTIPLE MINES PER IMAGE ====================
mine_count_configs = [
    (2, "double", 10),
    (3, "triple", 10),
    (4, "quad", 10)
]

for num_mines, mine_type, num_combinations in mine_count_configs:
    print(f"\nPhase {num_mines}: Generating {mine_type} mines per image...")

    # Generate random mine combinations
    mine_combinations = generate_mine_combinations(mine_images, num_mines, num_combinations)

    for terrain_idx, terrain_path in enumerate(terrains):
        print(f"  Processing terrain {terrain_idx + 1}/{len(terrains)}: {os.path.basename(terrain_path)}")

        terrain_img = cv2.imread(terrain_path)
        terrain_img = cv2.resize(terrain_img, FINAL_SIZE)

        # For each mine combination (REMOVED the variation loop)
        for combo_idx, selected_mines in enumerate(mine_combinations):
            # Create composite with selected mines
            composite, mine_annotations = overlay_specific_mines(terrain_img.copy(), selected_mines)

            # Base filename
            base_name = f"{os.path.basename(terrain_path)[:-4]}_{mine_type}_{combo_idx}"

            # Apply augmentations directly (REMOVED variation loop)
            # SINGLE AUGMENTATIONS
            single_configs = [
                ('lighting', change_lighting, (random.uniform(0.7, 1.3), random.uniform(0.8, 1.2)), None),
                ('motion_blur', motion_blur, (random.randint(3, 7),), None),
                ('gaussian_blur', gaussian_blur, (random.uniform(0, 1.5),), None),
                ('gaussian_noise', gaussian_noise, (random.uniform(0, 0.05),), None),
                ('salt_pepper', salt_pepper_noise, (random.uniform(0, 0.02),), None),
                ('perspective', perspective_transform, (random.uniform(10, 20), terrain_img), 'perspective'),
                ('rotation', rotate, (random.randint(0, 360), terrain_img), 'rotation'),
                ('scale', scale, (random.uniform(0.6, 1.4), terrain_img), 'scale'),
                ('hue_shift', hue_shift, (random.randint(-10, 10),), None),
            ]

            for i, (aug_name, aug_func, params, transform_type) in enumerate(single_configs):
                aug_img = aug_func(composite.copy(), *params)

                if transform_type is None:
                    aug_bboxes = mine_annotations.copy()
                else:
                    param_value = params[0]
                    aug_bboxes = transform_bbox_list(mine_annotations, transform_type, param_value, FINAL_SIZE[0],
                                                     FINAL_SIZE[1])

                filename = f"{base_name}_single{i}"
                cv2.imwrite(f"{OUTPUT_DIR}/{filename}.png", aug_img)
                save_yolo_annotation(aug_bboxes, filename)
                image_counter += 1

            # MULTI AUGMENTATIONS
            for m in range(MULTI_AUG_COUNT):
                img_multi = composite.copy()
                bboxes_multi = mine_annotations.copy()

                tilt_deg = random.uniform(10, 20)
                img_multi = perspective_transform(img_multi, tilt_deg, terrain_img)
                bboxes_multi = transform_bbox_list(bboxes_multi, 'perspective', tilt_deg, FINAL_SIZE[0],
                                                   FINAL_SIZE[1])

                angle = random.randint(0, 360)
                img_multi = rotate(img_multi, angle, terrain_img)
                bboxes_multi = transform_bbox_list(bboxes_multi, 'rotation', angle, FINAL_SIZE[0], FINAL_SIZE[1])

                scale_factor = random.uniform(0.6, 1.4)
                img_multi = scale(img_multi, scale_factor, terrain_img)
                bboxes_multi = transform_bbox_list(bboxes_multi, 'scale', scale_factor, FINAL_SIZE[0],
                                                   FINAL_SIZE[1])

                img_multi = change_lighting(img_multi, random.uniform(0.7, 1.3), random.uniform(0.8, 1.2))
                img_multi = motion_blur(img_multi, random.randint(3, 7))
                img_multi = gaussian_blur(img_multi, random.uniform(0, 1.5))
                img_multi = gaussian_noise(img_multi, random.uniform(0, 0.05))
                img_multi = salt_pepper_noise(img_multi, random.uniform(0, 0.02))
                img_multi = hue_shift(img_multi, random.randint(-10, 10))

                filename = f"{base_name}_multi{m}"
                cv2.imwrite(f"{OUTPUT_DIR}/{filename}.png", img_multi)
                save_yolo_annotation(bboxes_multi, filename)
                image_counter += 1

print(f"\nDataset generation complete!")
print(f"Generated {image_counter} images with annotations")
print(f"Images saved in: {OUTPUT_DIR}")
print(f"YOLO annotations saved in: {ANNOTATIONS_DIR}")

# Generate classes.txt for YOLO
with open("classes.txt", "w") as f:
    f.write("landmine\n")

print("Created classes.txt file")