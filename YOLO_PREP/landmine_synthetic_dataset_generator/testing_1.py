import cv2
import os

def visualize_yolo_labels(image_path, label_path, class_names):
    """
    Visualizes YOLO annotations on an image.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the YOLO label file (.txt).
        class_names (list): A list of class names corresponding to the class indices.
    """
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_width = float(parts[3]) * w
            box_height = float(parts[4]) * h

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = class_names[class_id]
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Labeled Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Make sure this list matches the order in your data.yaml file
    class_names = ['landmine']
    file_name = "asphalt_1_mine11_multi17"
    image_path=f"E:/dataset_img/output/{file_name}.png"
    label_path=f"E:/dataset_img/annotations/{file_name}.txt"
    # ---------------------
    visualize_yolo_labels(image_path, label_path, class_names)