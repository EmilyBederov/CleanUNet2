import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
from models.cleanspecnet import CleanSpecNet

class SpectrogramDataset(Dataset):
    def __init__(self, csv_path, sample_rate=16000, n_fft=1024, hop_length=256, win_length=1024):
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.spectrogram_fn = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
            power=1.0, normalized=True, center=False
        )
        
    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
        return waveform / waveform.abs().max()
    
    def __getitem__(self, index):
        noisy_path, clean_path = self.df.iloc[index]['noisy'], self.df.iloc[index]['clean']
        
        noisy = self._load_audio(noisy_path)
        clean = self._load_audio(clean_path)
        
        # Align lengths
        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]
        
        # Compute spectrograms
        noisy_spec = self.spectrogram_fn(noisy.unsqueeze(0))
        clean_spec = self.spectrogram_fn(clean.unsqueeze(0))
        
        return clean_spec, noisy_spec
    
    def __len__(self):
        return len(self.df)

def train_cleanspecnet():
    # Load config
    with open('configs/config_cleanspecnet.json') as f:
        config = json.load(f)
    
    network_config = config["network_config"]
    train_config = config["train_config"]
    trainset_config = config["trainset_config"]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = SpectrogramDataset('data/train_pairs.csv')
    val_dataset = SpectrogramDataset('data/val_pairs.csv')
    
    train_loader = DataLoader(train_dataset, batch_size=train_config["optimization"]["batch_size_per_gpu"], 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=train_config["optimization"]["batch_size_per_gpu"], 
                           shuffle=False, num_workers=4)
    
    # Create model
    model = CleanSpecNet(**network_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_config["optimization"]["learning_rate"])
    
    # Training loop
    model.train()
    for epoch in range(1000):  # Adjust as needed
        epoch_loss = 0
        for batch_idx, (clean_spec, noisy_spec) in enumerate(tqdm(train_loader)):
            clean_spec, noisy_spec = clean_spec.to(device), noisy_spec.to(device)
            
            optimizer.zero_grad()
            denoised_spec = model(noisy_spec)
            loss = F.l1_loss(denoised_spec, clean_spec)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {epoch_loss/len(train_loader):.6f}")
        
        # Save checkpoint every 100 epochs
        if epoch % 100 == 0:
            os.makedirs('checkpoints/cleanspecnet', exist_ok=True)
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, f'checkpoints/cleanspecnet/epoch_{epoch}.pkl')
    
    # Save final model
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, 'checkpoints/cleanspecnet/pretrained.pkl')

if __name__ == '__main__':
    train_cleanspecnet()
