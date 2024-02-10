import os
import cv2
import torch
from torch.utils.data import Dataset

class CustomObjectDetectionDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.label_dir = os.path.join(self.data_dir, 'labels')
        self.image_files = sorted([file for file in os.listdir(self.image_dir) if file.endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name.replace('.jpg', '.txt'))

        image = cv2.imread(image_path)
        labels = self.load_labels(label_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, labels

    def load_labels(self, label_path):
        labels = []
        with open(label_path, 'r') as file:
            lines = file.read().splitlines()
            for line in lines:
                values = line.split(',')
                if len(values) == 5:
                    x_min, y_min, x_max, y_max, class_id = map(float, values)
                    labels.append([x_min, y_min, x_max, y_max, class_id])
        return torch.tensor(labels, dtype=torch.float32)
