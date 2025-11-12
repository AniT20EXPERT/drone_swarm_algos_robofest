# # test existence of GPU
# import torch
# print(torch.cuda.is_available())  # should print True
# print(torch.cuda.get_device_name(0))  # should print your GPU name

# begin training
from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO('yolo11n.pt')
    results = model.train(data='dataset.yaml', epochs=100, imgsz=640, device=0, batch=38, workers=8)