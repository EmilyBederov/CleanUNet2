import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import json
import os
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from models.cleanspecnet import CleanSpecNet

class SpectrogramLoss(nn.Module):
    """
    Loss function for CleanSpecNet as described in the paper (Equation 1):
    1/Tspec * ||log(y/ŷ)||₁ + ||y - ŷ||_F / ||y||_F
    """
    def __init__(self):
        super(SpectrogramLoss, self).__init__()
    
    def forward(self, predicted_spec, target_spec):
        # Ensure no zeros for log computation
        eps = 1e-8
        predicted_spec = torch.clamp(predicted_spec, min=eps)
        target_spec = torch.clamp(target_spec, min=eps)
        
        # Log ratio loss: 1/Tspec * ||log(y/ŷ)||₁
        Tspec = target_spec.size(-1)  # time dimension
        log_ratio_loss = (1.0 / Tspec) * F.l1_loss(
            torch.log(target_spec / predicted_spec), 
            torch.zeros_like(target_spec)
        )
        
        # Frobenius norm loss: ||y - ŷ||_F / ||y||_F
        frobenius_loss = torch.norm(target_spec - predicted_spec, p='fro') / torch.norm(target_spec, p='fro')
        
        return log_ratio_loss + frobenius_loss

class VoiceBankSpectrogramDataset(Dataset):
    """
    Dataset for VoiceBank data in DNS format structure
    Expected structure:
    root/
    ├── training_set/
    │   ├── clean/
    │   └── noisy/
    └── testing_set/
        ├── clean/
        └── noisy/
    """
    def __init__(self, root_dir, subset='training', sample_rate=16000, n_fft=1024, 
                 hop_length=256, win_length=1024, crop_length_sec=10, add_noise=True):
        
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.crop_length = int(crop_length_sec * sample_rate)
        self.add_noise = add_noise
        
        # Set up paths based on subset
        if subset == 'training':
            self.clean_dir = self.root_dir / "training_set" / "clean"
            self.noisy_dir = self.root_dir / "training_set" / "noisy"
        else:
            self.clean_dir = self.root_dir / "testing_set" / "clean"
            self.noisy_dir = self.root_dir / "testing_set" / "noisy"
        
        # Get file pairs
        self.file_pairs = self._get_file_pairs()
        
        # STFT transform
        self.spectrogram_fn = T.Spectrogram(
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length,
            power=1.0,  # Linear magnitude spectrogram
            normalized=False,
            center=True
        )
        
        print(f"Loaded {len(self.file_pairs)} file pairs from {subset} set")
    
    def _get_file_pairs(self):
        """Get matching clean/noisy file pairs"""
        file_pairs = []
        
        if not self.clean_dir.exists() or not self.noisy_dir.exists():
            raise FileNotFoundError(f"Directories not found: {self.clean_dir} or {self.noisy_dir}")
        
        # Get all clean files
        clean_files = list(self.clean_dir.glob("*.wav"))
        
        for clean_file in clean_files:
            # Find corresponding noisy file
            noisy_file = self.noisy_dir / clean_file.name
            
            if noisy_file.exists():
                try:
                    # Quick validation
                    clean_info = torchaudio.info(str(clean_file))
                    noisy_info = torchaudio.info(str(noisy_file))
                    
                    if (clean_info.sample_rate == noisy_info.sample_rate and 
                        clean_info.num_frames > self.sample_rate):  # At least 1 second
                        file_pairs.append((clean_file, noisy_file))
                except:
                    continue
        
        return file_pairs
    
    def _load_and_process_audio(self, path):
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(path)
        
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Normalize
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform.squeeze(0)
    
    def _random_crop(self, waveform):
        """Randomly crop audio to specified length"""
        if len(waveform) <= self.crop_length:
            pad_length = self.crop_length - len(waveform)
            waveform = F.pad(waveform, (0, pad_length), mode='constant', value=0)
        else:
            start_idx = random.randint(0, len(waveform) - self.crop_length)
            waveform = waveform[start_idx:start_idx + self.crop_length]
        
        return waveform
    
    def _add_noise_augmentation(self, clean_waveform):
        """Add noise augmentation for training"""
        if not self.add_noise or random.random() > 0.3:
            return clean_waveform
        
        # Add gaussian noise with random SNR
        snr_db = random.uniform(10, 30)
        signal_power = torch.mean(clean_waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(clean_waveform) * torch.sqrt(noise_power)
        
        return clean_waveform + noise
    
    def __getitem__(self, index):
        clean_file, noisy_file = self.file_pairs[index]
        
        # Load clean and noisy audio
        clean_waveform = self._load_and_process_audio(clean_file)
        noisy_waveform = self._load_and_process_audio(noisy_file)
        
        # Align lengths and crop
        min_len = min(len(clean_waveform), len(noisy_waveform))
        clean_waveform = clean_waveform[:min_len]
        noisy_waveform = noisy_waveform[:min_len]
        
        clean_waveform = self._random_crop(clean_waveform)
        noisy_waveform = self._random_crop(noisy_waveform)
        
        # Optional additional noise augmentation
        if self.add_noise:
            noisy_waveform = self._add_noise_augmentation(noisy_waveform)
        
        # Compute spectrograms
        clean_spec = self.spectrogram_fn(clean_waveform)
        noisy_spec = self.spectrogram_fn(noisy_waveform)
        
        return noisy_spec, clean_spec
    
    def __len__(self):
        return len(self.file_pairs)

class CosineAnnealingWithWarmup:
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(self, optimizer, max_lr, total_steps, warmup_ratio=0.05):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.current_step = 0
    
    def step(self):
        if self.current_step < self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.max_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
        return lr

def train_cleanspecnet_voicebank():
    """Train CleanSpecNet on VoiceBank data"""
    
    parser = argparse.ArgumentParser(description='Train CleanSpecNet on VoiceBank')
    parser.add_argument('--config', type=str, default='configs/config_cleanspecnet.json',
                       help='Path to configuration file')
    parser.add_argument('--data_root', type=str, default='./voicebank_dns_format',
                       help='Path to VoiceBank data root')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = VoiceBankSpectrogramDataset(
        root_dir=args.data_root,
        subset='training',
        crop_length_sec=10,
        add_noise=True
    )
    
    val_dataset = VoiceBankSpectrogramDataset(
        root_dir=args.data_root,
        subset='testing', 
        crop_length_sec=10,
        add_noise=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    network_config = config["network_config"]
    model = CleanSpecNet(**network_config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} trainable parameters")
    
    # Loss function
    criterion = SpectrogramLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5
    )
    
    # Training setup
    total_iterations = 1000000  # 1M iterations as in paper
    start_iteration = 0
    
    # Resume from checkpoint if provided
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iteration = checkpoint.get('iteration', 0)
        print(f"Resumed from iteration {start_iteration}")
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWithWarmup(
        optimizer, 
        max_lr=2e-4, 
        total_steps=total_iterations,
        warmup_ratio=0.05
    )
    scheduler.current_step = start_iteration
    
    # Training directories
    os.makedirs('checkpoints/cleanspecnet_voicebank', exist_ok=True)
    os.makedirs('logs/cleanspecnet_voicebank', exist_ok=True)
    
    # Training loop
    print(f"Starting training from iteration {start_iteration}...")
    model.train()
    best_val_loss = float('inf')
    
    train_iter = iter(train_loader)
    progress_bar = tqdm(range(start_iteration, total_iterations), desc="Training")
    
    for iteration in progress_bar:
        try:
            noisy_spec, clean_spec = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            noisy_spec, clean_spec = next(train_iter)
        
        noisy_spec = noisy_spec.to(device)
        clean_spec = clean_spec.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predicted_spec = model(noisy_spec)
        loss = criterion(predicted_spec, clean_spec)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        current_lr = scheduler.step()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'LR': f'{current_lr:.2e}'
        })
        
        # Validation
        if (iteration + 1) % 1000 == 0:
            model.eval()
            val_loss = 0.0
            val_count = 0
            
            with torch.no_grad():
                for val_noisy, val_clean in val_loader:
                    val_noisy = val_noisy.to(device)
                    val_clean = val_clean.to(device)
                    
                    val_pred = model(val_noisy)
                    val_loss += criterion(val_pred, val_clean).item()
                    val_count += 1
                    
                    if val_count >= 20:  # Limit validation batches
                        break
            
            avg_val_loss = val_loss / val_count
            print(f"\nIteration {iteration + 1}: Validation Loss = {avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'iteration': iteration + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': config
                }, 'checkpoints/cleanspecnet_voicebank/best_model.pkl')
                print(f"Saved new best model with validation loss: {avg_val_loss:.6f}")
            
            model.train()
        
        # Save checkpoint
        if (iteration + 1) % 10000 == 0:
            torch.save({
                'iteration': iteration + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config
            }, f'checkpoints/cleanspecnet_voicebank/checkpoint_iter_{iteration + 1}.pkl')
            print(f"\nSaved checkpoint at iteration {iteration + 1}")
    
    # Save final model
    torch.save({
        'iteration': total_iterations,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config
    }, 'checkpoints/cleanspecnet_voicebank/final_model.pkl')
    
    print("CleanSpecNet training completed!")

if __name__ == '__main__':
    train_cleanspecnet_voicebank()