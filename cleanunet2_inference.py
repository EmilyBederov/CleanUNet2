import argparse
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
import time
from tqdm import tqdm
from models import CleanUNet2
import yaml
import json


def compute_stft(audio, n_fft=1024, hop_length=256, win_length=1024, power=1.0):
    """Compute Short-Time Fourier Transform of audio signal."""
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(audio.device),
        return_complex=True,
    )
    
    # Convert complex STFT to magnitude
    spec = torch.abs(stft) ** power
    
    return spec


def load_model(model_path, config_path, device):
    """Load the trained CleanUNet2 model."""
    
    # Load network configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    network_config = {
        "input_channels": config["network_config"]["cleanunet_input_channels"],
        "output_channels": config["network_config"]["cleanunet_output_channels"],
        "channels_H": config["network_config"]["cleanunet_channels_H"],
        "max_H": config["network_config"]["cleanunet_max_H"],
        "encoder_n_layers": config["network_config"]["cleanunet_encoder_n_layers"],
        "kernel_size": config["network_config"]["cleanunet_kernel_size"],
        "stride": config["network_config"]["cleanunet_stride"],
        "tsfm_n_layers": config["network_config"]["cleanunet_tsfm_n_layers"],
        "tsfm_n_head": config["network_config"]["cleanunet_tsfm_n_head"],
        "tsfm_d_model": config["network_config"]["cleanunet_tsfm_d_model"],
        "tsfm_d_inner": config["network_config"]["cleanunet_tsfm_d_inner"],
        "cleanspecnet_input_channels": config["network_config"]["cleanspecnet_input_channels"],
        "cleanspecnet_num_conv_layers": config["network_config"]["cleanspecnet_num_conv_layers"],
        "cleanspecnet_kernel_size": config["network_config"]["cleanspecnet_kernel_size"],
        "cleanspecnet_stride": config["network_config"]["cleanspecnet_stride"],
        "cleanspecnet_num_attention_layers": config["network_config"]["cleanspecnet_num_attention_layers"],
        "cleanspecnet_num_heads": config["network_config"]["cleanspecnet_num_heads"],
        "cleanspecnet_hidden_dim": config["network_config"]["cleanspecnet_hidden_dim"],
        "cleanspecnet_dropout": config["network_config"]["cleanspecnet_dropout"],
    }
    
    # Initialize model
    model = CleanUNet2(**network_config).to(device)
    
    # Load trained weights
    if model_path.endswith('.pth'):
        # Load the state dict directly if it's saved as a .pth file
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # If it's a .pkl checkpoint file, extract the model state dict
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def process_audio(model, audio_path, sample_rate, n_fft, hop_length, win_length, power, device):
    """Process a single audio file and return denoised audio and RTF."""
    
    # Load audio
    waveform, orig_sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if orig_sr != sample_rate:
        resampler = T.Resample(orig_sample_rate=orig_sr, new_sample_rate=sample_rate)
        waveform = resampler(waveform)
    
    # Ensure mono audio
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Move to device
    waveform = waveform.to(device)
    
    # Compute spectrogram
    spec = compute_stft(waveform, n_fft, hop_length, win_length, power)
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        denoised_audio, denoised_spec = model(waveform, spec)
    
    end_time = time.time()
    
    # Calculate RTF
    audio_duration = waveform.size(1) / sample_rate
    processing_time = end_time - start_time
    rtf = processing_time / audio_duration
    
    return denoised_audio.cpu(), rtf


def batch_process(model, input_dir, output_dir, sample_rate, n_fft, hop_length, win_length, power, device):
    """Process all audio files in a directory and report average RTF."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    rtf_values = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        audio_path = os.path.join(input_dir, audio_file)
        
        try:
            denoised_audio, rtf = process_audio(
                model, audio_path, sample_rate, n_fft, hop_length, win_length, power, device
            )
            
            # Save denoised audio
            output_path = os.path.join(output_dir, f"denoised_{audio_file}")
            torchaudio.save(output_path, denoised_audio, sample_rate)
            
            rtf_values.append(rtf)
            print(f"Processed {audio_file}: RTF = {rtf:.4f}")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    if rtf_values:
        avg_rtf = np.mean(rtf_values)
        min_rtf = np.min(rtf_values)
        max_rtf = np.max(rtf_values)
        
        print(f"\nRTF Statistics:")
        print(f"Average RTF: {avg_rtf:.4f}")
        print(f"Min RTF: {min_rtf:.4f}")
        print(f"Max RTF: {max_rtf:.4f}")
        
        # Save RTF stats
        stats_path = os.path.join(output_dir, "rtf_stats.json")
        stats = {
            "avg_rtf": float(avg_rtf),
            "min_rtf": float(min_rtf),
            "max_rtf": float(max_rtf),
            "individual_rtfs": {f: float(rtf) for f, rtf in zip(audio_files, rtf_values)}
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"RTF statistics saved to {stats_path}")


def process_single_file(model, audio_path, output_path, sample_rate, n_fft, hop_length, win_length, power, device):
    """Process a single audio file and save the result."""
    
    denoised_audio, rtf = process_audio(
        model, audio_path, sample_rate, n_fft, hop_length, win_length, power, device
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save denoised audio
    torchaudio.save(output_path, denoised_audio, sample_rate)
    
    print(f"Processed {os.path.basename(audio_path)}")
    print(f"RTF: {rtf:.4f}")
    print(f"Denoised audio saved to {output_path}")
    
    return rtf


def main():
    parser = argparse.ArgumentParser(description="CleanUNet2 Inference with RTF Measurement")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model configuration YAML file")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio file or directory")
    parser.add_argument("--output", type=str, required=True, help="Path to save output audio file or directory")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop length for STFT")
    parser.add_argument("--win_length", type=int, default=1024, help="Window length for STFT")
    parser.add_argument("--power", type=float, default=1.0, help="Power for spectrogram computation")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, args.config_path, device)
    print("Model loaded successfully")
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process single file
        process_single_file(
            model,
            args.input,
            args.output,
            args.sample_rate,
            args.n_fft,
            args.hop_length,
            args.win_length,
            args.power,
            device
        )
    else:
        # Process directory
        batch_process(
            model,
            args.input,
            args.output,
            args.sample_rate,
            args.n_fft,
            args.hop_length,
            args.win_length,
            args.power,
            device
        )


if __name__ == "__main__":
    main()
