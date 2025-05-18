import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from dataset import load_cleanunet2_dataset
from util import print_size
from util import LinearWarmupCosineDecay, save_checkpoint, load_checkpoint, prepare_directories_and_logger
from models import CleanUNet2
from stft_loss import MultiResolutionSTFTLoss, CleanUnetLoss, CleanUNet2Loss
import json
import os
import random
import numpy as np
from tqdm import tqdm


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def check_for_nan_and_inf(tensor, tensor_name="tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"{tensor_name} chas NaN values!")
    if torch.isinf(tensor).any():
        raise ValueError(f"{tensor_name} has Inf values!")
    
    
def validate(model, val_loader, loss_fn, iteration, trainset_config, logger, device):

    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():

        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating", leave=False)
        for i, (clean_audio, clean_spec, noisy_audio, noisy_spec) in progress_bar:
        # for i, (clean_audio, clean_spec, noisy_audio, noisy_spec) in enumerate(val_loader):
            clean_audio, clean_spec = clean_audio.to(device), clean_spec.to(device)
            noisy_audio, noisy_spec = noisy_audio.to(device), noisy_spec.to(device)

            denoised_audio, denoised_spec = model(noisy_audio, noisy_spec)  

            loss = loss_fn(clean_audio, denoised_audio)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # val_loss += loss
            val_loss += loss.item()

            num_batches += 1

        val_loss /= num_batches

    model.train()

    mel_transform = T.MelSpectrogram(
        sample_rate=trainset_config['sample_rate'],
        n_fft=trainset_config['n_fft'],
        win_length=trainset_config['win_length'],
        hop_length=trainset_config['hop_length'],
        n_mels=80
    ).to(device)
    amplitude_to_db = T.AmplitudeToDB(stype='power')
        
    #if rank == 0 and logger is not None:
    if logger is not None:
        print(f"Validation loss at iteration {iteration}: {val_loss:.6f}")

        # save to tensorboard
        logger.add_scalar("Validation/Loss", val_loss, iteration)

        num_samples = min(4, clean_spec.size(0))

        for i in range(num_samples):
            clean_audio_i = clean_audio[i].squeeze()
            denoised_audio_i = denoised_audio[i].squeeze()
            noisy_audio_i = noisy_audio[i].squeeze()

            clean_audio_np = clean_audio_i.cpu().numpy()
            denoised_audio_np = denoised_audio_i.cpu().numpy()
            noisy_audio_np = noisy_audio_i.cpu().numpy()

            clean_spec = amplitude_to_db(mel_transform(clean_audio_i))
            denoised_spec = amplitude_to_db(mel_transform(denoised_audio_i))
            noisy_spec = amplitude_to_db(mel_transform(noisy_audio_i))
            
            clean_spec_np = clean_spec.cpu().numpy()
            denoised_spec_np = denoised_spec.cpu().numpy()
            noisy_spec_np = noisy_spec.cpu().numpy()
                                                  
            # Plot spectrograms
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(clean_spec_np, origin='lower', aspect='auto')
            axs[0].set_title('Clean Spectrogram')
            axs[1].imshow(denoised_spec_np, origin='lower', aspect='auto')
            axs[1].set_title('Denoised Spectrogram')
            axs[2].imshow(noisy_spec_np, origin='lower', aspect='auto')
            axs[2].set_title('Noisy Spectrogram')
            plt.tight_layout()
            logger.add_figure('Spectrograms/Sample_{}'.format(i), fig, iteration)
            plt.close(fig)

            # Log audio samples to TensorBoard
            sample_rate = trainset_config['sample_rate']
            logger.add_audio('Audio/Clean_{}'.format(i), clean_audio_np, iteration, sample_rate=sample_rate)
            logger.add_audio('Audio/Denoised_{}'.format(i), denoised_audio_np, iteration, sample_rate=sample_rate)
            logger.add_audio('Audio/Noisy_{}'.format(i), noisy_audio_np, iteration, sample_rate=sample_rate)


def train(num_gpus, rank, group_name, exp_path, checkpoint_path, checkpoint_cleanunet_path, checkpoint_cleanspecnet_path, log, optimization, testloader, freeze_cleanspecnet=False, freeze_cleanunet=False, loss_config=None, device=None):
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create tensorboard logger.
    output_dir = os.path.join(log["directory"], exp_path)
    log_directory = os.path.join(output_dir, 'logs')
    ckpt_dir = os.path.join(output_dir, 'checkpoint')

    weight_decay = optimization["weight_decay"]
    learning_rate =  optimization["learning_rate"]
    max_norm = optimization["max_norm"]
    batch_size = optimization["batch_size_per_gpu"]
    iters_per_valid = log["iters_per_valid"]
    iters_per_ckpt = log["iters_per_ckpt"]

    logger = prepare_directories_and_logger(
        output_dir, log_directory, ckpt_dir, rank=0)

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)   

    import pandas as pd
    # Limit the dataset to only 5 samples
    df = pd.read_csv(trainset_config['csv_path'])
    temp_csv_path = "only_5_samples.csv"
    df.head(5).to_csv(temp_csv_path, index=False)

    trainloader = load_cleanunet2_dataset(
    csv_path=temp_csv_path,
    sample_rate=config['sample_rate'],
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    power=1.0,
    crop_length_sec=0.0,
    batch_size=1,  # safer to use batch size 1 to avoid batching conflicts
    num_workers=0
)
    print('Data loaded')

    # initialize the model
    model = CleanUNet2(**network_config).to(device)
    model.train()

    # apply gradient all reduce
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # load checkpoint
    global_step = 0
    latest_ckpt = None

    if checkpoint_path is not None and os.path.isdir(checkpoint_path):
        ckpt_files = glob.glob(os.path.join(checkpoint_path, '*.pkl'))
        if ckpt_files:
            latest_ckpt = max(ckpt_files, key=os.path.getctime)
            print(f"Resuming from latest checkpoint: {latest_ckpt}")
            model, optimizer, learning_rate, iteration = load_checkpoint(latest_ckpt, model, optimizer)
            global_step = iteration + 1
        else:
            print(f"No checkpoint files found in: {checkpoint_path}")
            print("Starting training from scratch.")
    else:
        print(f"No checkpoint directory found: {checkpoint_path}")
        print("Starting training from scratch.")
    if checkpoint_cleanunet_path is not None:
        if os.path.exists(checkpoint_cleanunet_path):
            print("Loading checkpoint '{}'".format(checkpoint_cleanunet_path))
            checkpoint_dict = torch.load(checkpoint_cleanunet_path, map_location='cpu')    
            new_checkpoint_dict = {}
            for k, v in checkpoint_dict['model_state_dict'].items():
                k = "clean_unet." + k
                new_checkpoint_dict[k] = v
            model.load_state_dict(new_checkpoint_dict, strict=False)
        else:
            print(f'No valid checkpoint model found at {checkpoint_cleanunet_path}.')
            exit()
    if checkpoint_cleanspecnet_path is not None:
        if os.path.exists(checkpoint_cleanspecnet_path):
            print("Loading checkpoint '{}'".format(checkpoint_cleanspecnet_path))
            checkpoint_dict = torch.load(checkpoint_cleanspecnet_path, map_location='cpu')    
            new_checkpoint_dict = {}
            for k, v in checkpoint_dict['state_dict'].items():
                k = "clean_spec_net." + k
                new_checkpoint_dict[k] = v
            model.load_state_dict(new_checkpoint_dict, strict=False)

        else:
            print(f'No valid checkpoint model found at {checkpoint_cleanspecnet_path}.')
            exit()

    if freeze_cleanspecnet:
        for param in model.clean_spec_net.parameters():
            param.requires_grad = False

    if freeze_cleanunet:
        for param in model.clean_unet.parameters():
            param.requires_grad = False

    print_size(model)

    # define learning rate scheduler
    scheduler = LinearWarmupCosineDecay(
                    optimizer,
                    lr_max=learning_rate,
                    n_iter=optimization["n_iters"],
                    iteration=global_step,
                    divider=25,
                    warmup_proportion=0.01,
                    phase=('linear', 'cosine'),
                )

    # define multi resolution stft loss
    if loss_config["stft_lambda"] > 0:
        mrstftloss = MultiResolutionSTFTLoss(**loss_config["stft_config"]).to(device)
    else:
        mrstftloss = None

    loss_fn = CleanUNet2Loss(**loss_config, mrstftloss=mrstftloss)

    # epoch = 1
    # print("Starting training...")
    # while global_step < optimization["n_iters"] + 1:    
    #     # for each epoch
    #     for step, (clean_audio, clean_spec, noisy_audio, noisy_spec) in enumerate(trainloader):

    #         noisy_audio = noisy_audio.to(device)
    #         noisy_spec = noisy_spec.to(device)
    #         clean_audio = clean_audio.to(device)
    #         clean_spec = clean_spec.to(device)

    #         # NaN and Inf tensor check
    #         try:
    #             check_for_nan_and_inf(noisy_audio, "noisy_audio")
    #             check_for_nan_and_inf(noisy_spec, "noisy_spec")
    #             check_for_nan_and_inf(clean_audio, "clean_audio")
    #             check_for_nan_and_inf(clean_spec, "clean_spec")
    #         except ValueError as e:
    #             print(e)
    #             continue

    #         optimizer.zero_grad()
    #         # forward propagation
    #         denoised_audio, denoised_spec = model(noisy_audio, noisy_spec)
    #         # calculate loss
    #         loss = loss_fn(clean_audio, denoised_audio)
    #         if num_gpus > 1:
    #             reduced_loss = reduce_tensor(loss.data, num_gpus).item()
    #         else:
    #             reduced_loss = loss.item()            
               
    #         # back-propagation
    #         loss.backward()

    #         if torch.isnan(loss).any():
    #             print("Loss contains NaN, terminating training")
    #             continue
    #             #break

    #         # gradient clipping
    #         grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    #         # update learning rate
    #         scheduler.step()            
    #         # update model parameters
    #         optimizer.step()

    #         print(f"Epoch: {epoch:<5} step: {step:<6} global step {global_step:<7} loss: {loss.item():.7f}", flush=True)

    #         if global_step > 0 and global_step % 10 == 0: 
    #             # save to tensorboard
    #             logger.add_scalar("Train/Train-Loss", reduced_loss, global_step)
    #             logger.add_scalar("Train/Gradient-Norm", grad_norm, global_step)
    #             logger.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], global_step)

    #         if global_step > 0 and global_step % iters_per_valid == 0 and rank == 0:
    #             validate(model, testloader, loss_fn, global_step, trainset_config, logger, device)
                
    #         # save checkpoint
    #         if global_step > 0 and global_step % iters_per_ckpt == 0 and rank == 0:
    #             checkpoint_name = '{}.pkl'.format(global_step)
    #             checkpoint_path = os.path.join(ckpt_dir, checkpoint_name)
    #             save_checkpoint(model, optimizer, learning_rate, global_step, checkpoint_path)
    #             print('model at iteration %s is saved' % global_step)
    #         global_step += 1
        
    #     epoch += 1

    # # After training, close TensorBoard.
    # if rank == 0:
    #     logger.close()

    # return 0

    total_steps = optimization["n_iters"]  # should be 50
    steps_per_epoch = 5
    epoch = 0
    total_epochs = total_steps // steps_per_epoch

    print("Starting training...")

    # while global_step < total_steps:
    #     progress_bar = tqdm(enumerate(trainloader), total=steps_per_epoch, desc=f"Epoch {epoch+1}", leave=False)
    #     for step, (clean_audio, clean_spec, noisy_audio, noisy_spec) in progress_bar:
    #     # for step, (clean_audio, clean_spec, noisy_audio, noisy_spec) in enumerate(trainloader):
    #         if step >= steps_per_epoch:
    #             break  # Only do 5 steps per epoch
            
    #         ########################################
    #         #change when running all over the data
    #         ########################################

    #         noisy_audio = noisy_audio.to(device)
    #         noisy_spec = noisy_spec.to(device)
    #         clean_audio = clean_audio.to(device)
    #         clean_spec = clean_spec.to(device)

    #         # NaN and Inf tensor check
    #         try:
    #             check_for_nan_and_inf(noisy_audio, "noisy_audio")
    #             check_for_nan_and_inf(noisy_spec, "noisy_spec")
    #             check_for_nan_and_inf(clean_audio, "clean_audio")
    #             check_for_nan_and_inf(clean_spec, "clean_spec")
    #         except ValueError as e:
    #             print(e)
    #             continue

    #         optimizer.zero_grad()
    #         # forward propagation
    #         denoised_audio, denoised_spec = model(noisy_audio, noisy_spec)

    #         # calculate loss
    #         loss = loss_fn(clean_audio, denoised_audio)
    #         reduced_loss = (
    #             reduce_tensor(loss.data, num_gpus).item()
    #             if num_gpus > 1 else loss.item()
    #         )

    #         # back-propagation
    #         loss.backward()
    #         progress_bar.set_postfix({"step": step, "loss": f"{reduced_loss:.4f}"})

    #         if torch.isnan(loss).any():
    #             print("Loss contains NaN, terminating training")
    #             continue

    #         # gradient clipping
    #         grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    #         # update learning rate and parameters
    #         scheduler.step()
    #         optimizer.step()

    #         print(f"Epoch: {epoch:<5} step: {step:<6} global step {global_step:<7} loss: {loss.item():.7f}", flush=True)

    #         if global_step > 0 and global_step % 10 == 0:
    #             logger.add_scalar("Train/Train-Loss", reduced_loss, global_step)
    #             logger.add_scalar("Train/Gradient-Norm", grad_norm, global_step)
    #             logger.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], global_step)

    #         if global_step > 0 and global_step % iters_per_ckpt == 0 and rank == 0:
    #             checkpoint_name = '{}.pkl'.format(global_step)
    #             checkpoint_file = os.path.join(ckpt_dir, checkpoint_name)
    #             save_checkpoint(model, optimizer, learning_rate, global_step, checkpoint_file)
    #             print(f'Model at iteration {global_step} is saved')

    #         global_step += 1

    #         if global_step >= total_steps:
    #             break  #  Exit outer loop early if total steps exceeded

    #     # Run validation at the end of each epoch
    #     if rank == 0:
    #         print(f"Epoch {epoch + 1} finished. Running validation...")
    #         model.eval()
    #         validate(model, testloader, loss_fn, global_step, trainset_config, logger, device)
    #         model.train()

    #     epoch += 1

    # if rank == 0:
    #     logger.close()

    #     final_model_path = os.path.join(ckpt_dir, 'final_model.pth')
    #     torch.save(model.state_dict(), final_model_path)
    #     print(f"Final model weights saved for inference: {final_model_path}")

    # return 0
   
    print("Starting 5-sample training...")

    model.train()
    for step, (clean_audio, clean_spec, noisy_audio, noisy_spec) in enumerate(trainloader):
        if step >= 5:
            break  # only process 5 samples

        noisy_audio = noisy_audio.to(device)
        noisy_spec = noisy_spec.to(device)
        clean_audio = clean_audio.to(device)
        clean_spec = clean_spec.to(device)

        try:
            check_for_nan_and_inf(noisy_audio, "noisy_audio")
            check_for_nan_and_inf(noisy_spec, "noisy_spec")
            check_for_nan_and_inf(clean_audio, "clean_audio")
            check_for_nan_and_inf(clean_spec, "clean_spec")
        except ValueError as e:
            print(e)
            continue

        optimizer.zero_grad()
        denoised_audio, denoised_spec = model(noisy_audio, noisy_spec)
        loss = loss_fn(clean_audio, denoised_audio)

        if torch.isnan(loss).any():
            print("Loss contains NaN, skipping step")
            continue

        loss.backward()
        optimizer.step()

        print(f"[Step {step}] Loss: {loss.item():.6f}")

    # Save model for inference
    if rank == 0:
        os.makedirs("output", exist_ok=True)
        torch.save(model.state_dict(), "output/cleanunet2_5sample.pth")
        print("[✓] Final model weights saved at: output/cleanunet2_5sample.pth")
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config.json', 
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    with open("configs/config.json") as f:
        data = f.read()
    config = json.loads(data)

    train_config            = config["train_config"]        # training parameters
    global dist_config
    dist_config             = config["dist_config"]         # to initialize distributed training
    global network_config
    network_config          = config["network_config"]      # to define network
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("DEBUG: Config keys loaded:", list(config.keys()))

    global trainset_config
    trainset_config = config["trainset"]     # to load trainset
    # Extend trainset_config so validate() has what it needs
    trainset_config["sample_rate"] = config["sample_rate"]
    trainset_config["n_fft"] = 1024
    trainset_config["win_length"] = 1024
    trainset_config["hop_length"] = 256
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")
    
    testloader = load_cleanunet2_dataset(
        csv_path=config['valset']['csv_path'],  #  make sure this path points to the validation CSV
        sample_rate=config['sample_rate'],
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        power=1.0,
        crop_length_sec=0.0,
        batch_size=config['batch_size'],           # or use a different val batch size if you want
        num_workers=config['num_workers']
    )
    print('Validation set loaded')
    
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True

    # train(num_gpus, args.rank, args.group_name, **train_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(
    num_gpus,
    args.rank,
    args.group_name,
    train_config["exp_path"],
    train_config["checkpoint_path"],
    train_config["checkpoint_cleanunet_path"],
    train_config["checkpoint_cleanspecnet_path"],
    train_config["log"],
    train_config["optimization"],
    testloader=testloader,   
    freeze_cleanspecnet=train_config.get("freeze_cleanspecnet", False),
    freeze_cleanunet=train_config.get("freeze_cleanunet", False),
    loss_config=train_config.get("loss_config", None),
    device=device
)


    
