import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class FacialRatioDataset(Dataset):
    def __init__(self, csv_path, device):
        self.data = pd.read_csv(csv_path)
        self.device = device
        
        # Convert gender to one-hot (0 for male, 1 for female)
        self.data['gender_encoded'] = (self.data['gender'] == 'Female').astype(int)
        
        # Features: gender_encoded, CJWR, WHR, ES, LF_FH, FW_LFH
        self.features = self.data[['gender_encoded', 'CJWR', 'WHR', 'ES', 'LF_FH', 'FW_LFH']].values
        
        # Target: BMI
        self.targets = self.data['bmi'].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx]).to(self.device)
        target = torch.FloatTensor([self.targets[idx]]).to(self.device)
        return features, target

def get_dataloader(csv_path, batch_size=32, shuffle=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    dataset = FacialRatioDataset(csv_path, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    dataloader = get_dataloader('merged_data.csv')
    for features, target in dataloader:
        print(features, target)
        break