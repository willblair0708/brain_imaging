"""import functions and libraries"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Custom dataset class for handling NIfTI files
class BrainDataset(Dataset):
    def __init__(self, t1w_files, t2w_files, fa_files, adc_files, transform=None):
        self.t1w_files = t1w_files
        self.t2w_files = t2w_files
        self.fa_files = fa_files
        self.adc_files = adc_files
        self.transform = transform

    def __len__(self):
        return len(self.t1w_files)

    def __getitem__(self, idx):
        t1w_image = nib.load(self.t1w_files[idx]).get_fdata()
        t2w_image = nib.load(self.t2w_files[idx]).get_fdata()
        fa_image = nib.load(self.fa_files[idx]).get_fdata()
        adc_image = nib.load(self.adc_files[idx]).get_fdata()

        input_image = np.stack([t1w_image, t2w_image], axis=0)
        target_image = np.stack([fa_image, adc_image], axis=0)

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc = self.encoder(x)
        middle = self.middle(enc)
        dec = self.decoder(middle)
        out = self.final(dec)
        return out

# Function for initializing the U-Net model
def initialize_unet(in_channels, out_channels):
    return UNet(in_channels, out_channels)

# Function for training the U-Net model
def train_unet(unet, dataloader, device, epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)

    unet.train()
    unet.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = unet(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    print("Training completed.")

if __name__ == "__main__":
    data_dir = os.path.join('..', 'data')

    # Replace these lists with the paths to your training dataset
    t1w_files = []
    t2w_files = []
    fa_files = []
    adc_files = []

    dataset = BrainDataset(t1w_files, t2w_files, fa_files, adc_files, transform=Compose([torch.tensor]))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unet = initialize_unet(in_channels=2, out_channels=2)
    train_unet(unet, dataloader, device, epochs=100, learning_rate=0.001)

    # Save the trained model
    torch.save(unet.state_dict(), os.path.join(data_dir, "unet_model.pth"))

    print("Model saved.")

