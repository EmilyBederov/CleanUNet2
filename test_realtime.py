import time
import os
import torch
import torchaudio
import torchaudio.transforms as T
from models import CleanUNet2
#
# --- CONFIG ---
wav_path = "data/noisy/p234_140.wav"
output_path = "data/denoised/sample_denoised.wav"
model_path = "output/cleanunet2_5sample.pth"
sample_rate = 16000

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CleanUNet2(
    cleanunet_input_channels=1,
    cleanunet_output_channels=1,
    cleanunet_channels_H=32,
    cleanunet_max_H=256,
    cleanunet_encoder_n_layers=5,
    cleanunet_kernel_size=4,
    cleanunet_stride=2,
    cleanunet_tsfm_n_layers=2
).to(device)

state = torch.load(model_path, map_location=device)
model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
model.eval()

# --- Load audio ---
waveform, sr = torchaudio.load(wav_path)
if sr != sample_rate:
    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

waveform = waveform.unsqueeze(0).to(device)  # [1, 1, T]
duration = waveform.shape[-1] / sample_rate

# --- Compute spectrogram (same config used in training)
spec_transform = T.Spectrogram(
    n_fft=512,
    hop_length=128,
    win_length=512,
    power=2,
    normalized=True,
    center=False
).to(device)

spectrogram = spec_transform(waveform.squeeze(0))  # [1, F, T]
spectrogram = spectrogram.unsqueeze(0)  # [1, 1, F, T]

# --- Warm-up
with torch.no_grad():
    _ = model(waveform, spectrogram)

# --- Timed inference
start = time.time()
with torch.no_grad():
    denoised = model(waveform, spectrogram)
torch.cuda.synchronize() if device.type == "cuda" else None
end = time.time()

# --- Save output
denoised = denoised.squeeze().cpu().clamp(-1, 1)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
torchaudio.save(output_path, denoised.unsqueeze(0), sample_rate)

# --- RTF
elapsed = end - start
rtf = elapsed / duration
print(f"✅ Saved to: {output_path}")
print(f"🕒 Inference time: {elapsed:.4f} sec")
print(f"📏 Audio duration: {duration:.2f} sec")
print(f"⚡ Real-Time Factor (RTF): {rtf:.4f} → {'✅ Real-time' if rtf < 1 else '❌ Too slow'}")