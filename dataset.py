import os
import torch
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class PairedAudioDataset(Dataset):
    def __init__(self, csv_path, sample_rate=16000, n_fft=1024, hop_length=256, win_length=1024, power=1.0, crop_length_sec=0.0):
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.crop_length = int(crop_length_sec * sample_rate)
        self.spectrogram_fn = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=power, normalized=True, center=False)

         # LOAD THE CSV INTO self.files
        self.files = list(zip(self.df["noisy"], self.df["clean"]))

    @staticmethod
    def _load_audio_and_resample(filepath: str, target_sr: int = 16000) -> torch.Tensor:
        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception as e:
            print(f" Failed to load {filepath}: {e}")
            return torch.zeros(target_sr)  # return 1 second of silence

        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        waveform = waveform / waveform.abs().max()

        return waveform

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
        return waveform / waveform.abs().max()

    def __getitem__(self, index):
        noisy_path, clean_path = self.files[index]

        noisy = self._load_audio_and_resample(noisy_path, self.sample_rate)
        clean = self._load_audio_and_resample(clean_path, self.sample_rate)

        # Crop (optional)
        if self.crop_length > 0:
            crop_len = min(len(noisy), len(clean))
            if self.crop_length < crop_len:
                start = torch.randint(0, crop_len - self.crop_length + 1, (1,)).item()
                noisy = noisy[start:start + self.crop_length]
                clean = clean[start:start + self.crop_length]

        # Align lengths
        min_len = min(len(noisy), len(clean))
        noisy = noisy[:min_len]
        clean = clean[:min_len]

        # Pad to valid STFT length
        n_fft = self.spectrogram_fn.n_fft
        hop = self.spectrogram_fn.hop_length
        total_len = ((min_len - n_fft) // hop + 1) * hop + n_fft
        pad = max(0, total_len - min_len)
        noisy = F.pad(noisy, (0, pad))
        clean = F.pad(clean, (0, pad))

        # Compute spectrograms
        noisy_spec = self.spectrogram_fn(noisy.unsqueeze(0))  # → [F, T]
        clean_spec = self.spectrogram_fn(clean.unsqueeze(0))  # → [F, T]

        return clean.unsqueeze(0), clean_spec, noisy.unsqueeze(0), noisy_spec

    def __len__(self):
        return len(self.df)

def collate_fn(batch):
    clean_audios, clean_specs, noisy_audios, noisy_specs = zip(*batch)

    # Pad audios to max length in batch
    max_len = max(a.shape[-1] for a in clean_audios)
    
    # Define pad1d function here
    def pad1d(t, L): 
        return F.pad(t, (0, L - t.shape[-1]))

    clean_audio_batch = torch.stack([pad1d(x, max_len) for x in clean_audios])
    noisy_audio_batch = torch.stack([pad1d(x, max_len) for x in noisy_audios])

    # Pad spectrograms to max T
    max_T = max(s.shape[-1] for s in clean_specs)
    
    # Define pad2d function here
    def pad2d(t, T): 
        return F.pad(t, (0, T - t.shape[-1]))

    clean_spec_batch = torch.stack([pad2d(x, max_T) for x in clean_specs])  # [B, F, T]
    noisy_spec_batch = torch.stack([pad2d(x, max_T) for x in noisy_specs])  # [B, F, T]

    return clean_audio_batch, clean_spec_batch, noisy_audio_batch, noisy_spec_batch

def load_cleanunet2_dataset(csv_path, sample_rate, n_fft, hop_length, win_length, power, crop_length_sec, batch_size, num_workers):
    dataset = PairedAudioDataset(csv_path=csv_path, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                 win_length=win_length, power=power, crop_length_sec=crop_length_sec)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    return loader