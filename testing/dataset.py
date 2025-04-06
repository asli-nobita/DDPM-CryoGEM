import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

class CryoGEMDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.image_files = sorted([f for f in self.root_dir.glob("*.png")])
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        # Create a dummy mask (all ones) since the model expects 4 channels
        mask = torch.ones_like(image)
        
        # Combine image and mask
        combined = torch.cat([image, mask], dim=0)
        
        return combined

def get_data_loader(root_dir, batch_size=32, shuffle=True, num_workers=4):
    dataset = CryoGEMDataset(root_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

if __name__ == "__main__":
    # Test the dataset
    dataset = CryoGEMDataset("testing/data/Ribosome(10028)/real_data")
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")  # Should be [2, 128, 128] (image + mask)
    
    # Test the dataloader
    dataloader = get_data_loader("testing/data/Ribosome(10028)/real_data", batch_size=4)
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")  # Should be [4, 2, 128, 128]
        break 