import torch
from torch.utils.data import Dataset
import cv2
import os
from glob import glob
import numpy as np
import pandas as pd
import torchvision

class ImageDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        """
        Args:
            dataset_path (str): Path to the dataset folder containing images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = glob(os.path.join(dataset_path, "aligned/*.bmp"))
        self.image_idx_map = {os.path.basename(path): path for idx, path in enumerate(self.image_paths)}
        print(self.image_idx_map)
        self.training_data_csv = pd.read_csv(os.path.join(dataset_path, "trainingset.csv"))
        self.normalize_data()
    
    def normalize_data(self):
        self.training_data_csv["bmi"] = (self.training_data_csv["bmi"] - np.mean(self.training_data_csv["bmi"])) / np.std(self.training_data_csv["bmi"])

    def __len__(self):
        return len(self.training_data_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_idx = self.training_data_csv.iloc[idx]["name"]
        image_path = self.image_idx_map[image_idx]
        bmi = self.training_data_csv.loc[idx, "bmi"]
        bmi = torch.tensor(bmi, dtype=torch.float32)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (32, 32))
        image = image.astype(np.float32) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to tensor if not already done by transforms
        if not torch.is_tensor(image):
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)  # Convert from HWC to CHW format
            
        return (image, bmi)

def get_dataloader(dataset_path, batch_size=32, shuffle=True, num_workers=4, transform=None):
    """
    Creates and returns a DataLoader for the image dataset
    
    Args:
        dataset_path (str): Path to dataset directory
        batch_size (int): How many samples per batch to load
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): How many subprocesses to use for data loading
        transform (callable, optional): Optional transform to be applied on the samples
    """
    dataset = ImageDataset(dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
