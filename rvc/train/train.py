import os
import re
import sys
import glob
import json
import torch
import datetime

import math
from typing import Tuple
import itertools

from collections import deque
from distutils.util import strtobool
from random import randint, shuffle
from time import time as ttime
from time import sleep
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.nn import functional as F

from torch.nn.utils import clip_grad_norm_
clip_grad_norm_ = torch.nn.utils.clip_grad_norm_

import torch.distributed as dist
import torch.multiprocessing as mp

#from accelerate import Accelerator # WIP.

# Custom optimizers:
from custom_optimizers.ranger import Ranger
now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import rvc.lib.zluda

from utils import (
    HParams,
    plot_spectrogram_to_numpy,
    summarize,
    load_checkpoint,
    save_checkpoint,
    latest_checkpoint_path,
    load_wav_to_torch,
)

from losses import discriminator_loss, feature_loss, generator_loss, kl_loss

from mel_processing import mel_spectrogram_torch, spec_to_mel_torch, MultiScaleMelSpectrogramLoss

from rvc.train.process.extract_model import extract_model
from rvc.lib.algorithm import commons

# MultiPeriodDiscriminatorV2 ( Original Discriminator - A must have. )
#from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator


# Parse command line arguments
model_name = sys.argv[1]
save_every_epoch = int(sys.argv[2])
total_epoch = int(sys.argv[3])
pretrainG = sys.argv[4]
pretrainD = sys.argv[5]
version = sys.argv[6]
gpus = sys.argv[7]
batch_size = int(sys.argv[8])
sample_rate = int(sys.argv[9])
pitch_guidance = strtobool(sys.argv[10])
save_only_latest = strtobool(sys.argv[11])
save_every_weights = strtobool(sys.argv[12])
cache_data_in_gpu = strtobool(sys.argv[13])
use_warmup = strtobool(sys.argv[14])
warmup_duration = int(sys.argv[15])
cleanup = strtobool(sys.argv[16])
vocoder = sys.argv[17]
n_value = int(sys.argv[18]) #

current_dir = os.getcwd()
experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "sliced_audios")

with open(config_save_path, "r") as f:
    config = json.load(f)

config = HParams(**config)
config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

# for nVidia's CUDA device selection can be done from command line / UI
# for AMD the device selection can only be done from .bat file using HIP_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = gpus.replace("-", ",")

# Torch backends config
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Globals
global_step = 0
mini_batches = n_value
warmup_epochs = warmup_duration
warmup_enabled = use_warmup
warmup_completed = False


# --------------------------   Custom functions land in here   --------------------------


# Mel spectrogram similarity metric ( Predicted ∆ Real ) using L1 loss
def mel_spectrogram_similarity(y_hat_mel, y_mel):
    if y_hat_mel.shape != y_mel.shape:
        trimmed_shape = tuple(min(dim_a, dim_b) for dim_a, dim_b in zip(y_hat_mel.shape, y_mel.shape))
        y_hat_mel = y_hat_mel[..., :trimmed_shape[-1]]
        y_mel = y_mel[..., :trimmed_shape[-1]]
    
    # Calculate the L1 loss between the generated mel and original mel spectrograms
    loss_mel = F.l1_loss(y_hat_mel, y_mel)

    # Convert the L1 loss to a similarity score between 0 and 100
    # Scale the loss to a percentage, where a perfect match (0 loss) gives 100% similarity
    mel_spec_similarity = 100.0 - (loss_mel.item() * 100.0)

    # Convert the similarity percentage to a tensor
    mel_spec_similarity = torch.tensor(mel_spec_similarity)

    # Clip the similarity percentage to ensure it stays within the desired range
    mel_spec_similarity = torch.clamp(mel_spec_similarity, min=0.0, max=100.0)

    return mel_spec_similarity

# --------------------------   Custom functions End here   --------------------------



# --------------------------   Execution   --------------------------

import logging
logging.getLogger("torch").setLevel(logging.ERROR)


class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"Current time: {current_time} | Time per epoch: {elapsed_time_str}"


def main():
    """
    Main function to start the training process.
    """

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "50000"
    # Check sample rate
    wavs = glob.glob(
        os.path.join(os.path.join(experiment_dir, "sliced_audios"), "*.wav")
    )
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != sample_rate:
            print(
                f"Error: Pretrained model sample rate ({sample_rate} Hz) does not match dataset audio sample rate ({sr} Hz)."
            )
            os._exit(1)
    else:
        print("No wav file found.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        n_gpus = 1
    else:
        device = torch.device("cpu")
        n_gpus = 1
        print("No GPU detected, fallback to CPU. This will take a very long time...")

    def start():
        """
        Starts the training process with multi-GPU support or CPU.
        """
        children = []
        pid_data = {"process_pids": []}
        with open(config_save_path, "r") as pid_file:
            try:
                existing_data = json.load(pid_file)
                pid_data.update(existing_data)
            except json.JSONDecodeError:
                pass
        with open(config_save_path, "w") as pid_file:
            for i in range(n_gpus):
                subproc = mp.Process(
                    target=run,
                    args=(
                        i,
                        n_gpus,
                        experiment_dir,
                        pretrainG,
                        pretrainD,
                        pitch_guidance,
                        total_epoch,
                        save_every_weights,
                        config,
                        device,
                    ),
                )
                children.append(subproc)
                subproc.start()
                pid_data["process_pids"].append(subproc.pid)
            json.dump(pid_data, pid_file, indent=4)

        for i in range(n_gpus):
            children[i].join()



    if cleanup:
        print("Removing files from the previous training attempt...")

        # Clean up unnecessary files
        for root, dirs, files in os.walk(
            os.path.join(now_dir, "logs", model_name), topdown=False
        ):
            for name in files:
                file_path = os.path.join(root, name)
                file_name, file_extension = os.path.splitext(name)
                if (
                    file_extension == ".0"
                    or (file_name.startswith("D_") and file_extension == ".pth")
                    or (file_name.startswith("G_") and file_extension == ".pth")
                    or (file_name.startswith("added") and file_extension == ".index")
                ):
                    os.remove(file_path)
            for name in dirs:
                if name == "eval":
                    folder_path = os.path.join(root, name)
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                    os.rmdir(folder_path)

        print("Cleanup done!")

    start()


def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
    pitch_guidance,
    custom_total_epoch,
    custom_save_every_weights,
    config,
    device,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrainG (str): Path to the pre-trained generator model.
        pretrainD (str): Path to the pre-trained discriminator model.
        pitch_guidance (bool): Flag indicating whether to use pitch guidance during training.
        custom_total_epoch (int): The total number of epochs for training.
        custom_save_every_weights (int): The interval (in epochs) at which to save model weights.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step, warmup_completed


    if 'warmup_completed' not in globals():
        warmup_completed = False

    if rank == 0 and warmup_enabled:
        print(f"//////  Warmup enabled. Training will gradually increase learning rates over {warmup_epochs} epochs.  //////")

    if rank == 0:
        writer = SummaryWriter(log_dir=experiment_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(experiment_dir, "eval"))
    else:
        writer, writer_eval = None, None

    dist.init_process_group(
        backend="gloo", #"nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Create datasets and dataloaders
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid,
    )

    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    collate_fn = TextAudioCollateMultiNSFsid()
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # Initialize models and optimizers
    from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
    from rvc.lib.algorithm.synthesizers import Synthesizer

    # Initialize models and optimizers
    net_g = Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0=pitch_guidance == True,  # converting 1/0 to True/False
        is_half=config.train.fp16_run and device.type == "cuda",
        sr=sample_rate,
        vocoder=vocoder
    ).to(device)



    net_d = MultiPeriodDiscriminator(config.model.use_spectral_norm).to(device)

    optim_g = Ranger(
        net_g.parameters(),
        lr = 0.0001, # config.train.learning_rate,
        betas = (0.8, 0.99), # config.train.betas,
        eps = 1e-8, # config.train.eps,
        # Ranger params:
        weight_decay = 0,
        alpha=0.5,
        k=6,
        N_sma_threshhold=5, # 4 or 5 can be tried
        use_gc=False,
        gc_conv_only=False,
        gc_loc=False,
    )
    optim_d = Ranger(
        net_d.parameters(),
        lr = 1e-4, # config.train.learning_rate,
        betas = (0.8, 0.99), # config.train.betas,
        eps = 1e-8, # config.train.eps,
        # Ranger params:
        weight_decay = 0,
        alpha=0.5,
        k=6,
        N_sma_threshhold=5, # 4 or 5 can be tried
        use_gc=False,
        gc_conv_only=False,
        gc_loc=False,
    )

    fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)

    # Wrap models with DDP for multi-gpu processing
    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[rank]) # find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[rank]) # find_unused_parameters=True)
    else:
        # CPU only (no need to move as they're already on CPU)
        net_g = net_g
        net_d = net_d

    # Load checkpoint if available
    try:
        print("Starting training...")
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "D_*.pth"), net_d, optim_d
        )
        _, _, _, epoch_str = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "G_*.pth"), net_g, optim_g
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)

    except:
        epoch_str = 1
        global_step = 0

    # Loading the pretrained Generator model
        if pretrainG != "" and pretrainG != "None":
            if rank == 0:
                print(f"Loaded pretrained (G) '{pretrainG}'")
            if hasattr(net_g, "module"):
                net_g.module.load_state_dict(
                    torch.load(pretrainG, map_location="cpu")["model"]
                )
            else:
                net_g.load_state_dict(
                    torch.load(pretrainG, map_location="cpu")["model"]
                )


    # Loading the pretrained Discriminator model
        if pretrainD != "" and pretrainD != "None":
            if rank == 0:
                print(f"Loaded pretrained (D) '{pretrainD}'")
            if hasattr(net_d, "module"):
                net_d.module.load_state_dict(
                    torch.load(pretrainD, map_location="cpu")["model"]
                )
            else:
                net_d.load_state_dict(
                    torch.load(pretrainD, map_location="cpu")["model"]
                )


    # Initialize the warmup scheduler only if `warmup_enabled` is True
    if warmup_enabled:
        # Warmup for: Generator
        warmup_scheduler_g = torch.optim.lr_scheduler.LambdaLR(
            optim_g,
            lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
        )
        # Warmup for: MPD
        warmup_scheduler_d = torch.optim.lr_scheduler.LambdaLR(
            optim_d,
            lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
        )

    # Ensure initial_lr is set when warmup_enabled is False
    if not warmup_enabled:
        # For: Generator
        for param_group in optim_g.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']
        # For: Discriminator
        for param_group in optim_d.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']


    # For the decay phase (after warmup)
        # For: Generator
    decay_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.train.lr_decay, last_epoch=epoch_str - 1
    )
        # For: Discriminator
    decay_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.train.lr_decay, last_epoch=epoch_str - 1
    )

    scaler = GradScaler(enabled=config.train.fp16_run and device.type == "cuda")

    cache = []
    # get the first sample as reference for tensorboard evaluation
    # custom reference temporarily disabled
    if True == False and os.path.isfile(
        os.path.join("logs", "reference", f"ref{sample_rate}.wav")
    ):
        phone = np.load(
            os.path.join("logs", "reference", f"ref{sample_rate}_feats.npy")
        )
        # expanding x2 to match pitch size
        phone = np.repeat(phone, 2, axis=0)
        phone = torch.FloatTensor(phone).unsqueeze(0).to(device)
        phone_lengths = torch.LongTensor(phone.size(0)).to(device)
        pitch = np.load(os.path.join("logs", "reference", f"ref{sample_rate}_f0c.npy"))
        # removed last frame to match features
        pitch = torch.LongTensor(pitch[:-1]).unsqueeze(0).to(device)
        pitchf = np.load(os.path.join("logs", "reference", f"ref{sample_rate}_f0f.npy"))
        # removed last frame to match features
        pitchf = torch.FloatTensor(pitchf[:-1]).unsqueeze(0).to(device)
        sid = torch.LongTensor([0]).to(device)
        reference = (
            phone,
            phone_lengths,
            pitch if pitch_guidance else None,
            pitchf if pitch_guidance else None,
            sid,
        )
    else:
        for info in train_loader:
            phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
            reference = (
                phone.to(device),
                phone_lengths.to(device),
                pitch.to(device) if pitch_guidance else None,
                pitchf.to(device) if pitch_guidance else None,
                sid.to(device),
            )
            break

    for epoch in range(epoch_str, total_epoch + 1):
        train_and_evaluate(
            rank,
            epoch,
            config,
            [net_g, net_d],
            [optim_g, optim_d],
            scaler,
            [train_loader, None],
            [writer, writer_eval],
            cache,
            custom_save_every_weights,
            custom_total_epoch,
            device,
            reference,
            fn_mel_loss,
        )

        if warmup_enabled and epoch <= warmup_epochs:
            # Starts the warmup phase if warmup_epochs =/= warmup_epochs
            warmup_scheduler_g.step()
            warmup_scheduler_d.step()

            # Logging of finished warmup
            if epoch == warmup_epochs:
                warmup_completed = True
                print(f"//////  Warmup completed at warmup epochs:{warmup_epochs}  //////")
                # Gen:
                print(f"//////  LR G: {optim_g.param_groups[0]['lr']}  //////")
                # Discs:
                print(f"//////  LR D: {optim_d.param_groups[0]['lr']}  //////")
                # Decay gamma:
                print(f"//////  Starting the exponential lr decay with gamma of {config.train.lr_decay}  //////")
 
        # Once the warmup phase is completed, uses exponential lr decay
        if not warmup_enabled or warmup_completed:
            decay_scheduler_g.step()
            decay_scheduler_d.step()


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    scaler,
    loaders,
    writers,
    cache,
    custom_save_every_weights,
    custom_total_epoch,
    device,
    reference,
    fn_mel_loss,
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        hps (Namespace): Hyperparameters.
        nets (list): List of models [net_g, net_d].
        optims (list): List of optimizers [optim_g, optim_d].
        scaler (GradScaler): Gradient scaler for mixed precision training.
        loaders (list): List of dataloaders [train_loader, eval_loader].
        writers (list): List of TensorBoard writers [writer, writer_eval].
        cache (list): List to cache data in GPU memory.
        use_cpu (bool): Whether to use CPU for training.
    """
    global global_step, warmup_completed


    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader = loaders[0] if loaders is not None else None
    if writers is not None:
        writer = writers[0]

    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    # Data caching
    if device.type == "cuda" and cache_data_in_gpu:
        data_iterator = cache
        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                # phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid
                info = [tensor.cuda(rank, non_blocking=True) for tensor in info]
                cache.append((batch_idx, info))
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()

    # Over N mini-batches loss averaging
    N = mini_batches  # Number of mini-batches after which the loss is logged
    running_loss_gen = 0.0  # Running loss for generator
    running_loss_disc = 0.0  # Running loss for discriminator

    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            if device.type == "cuda" and not cache_data_in_gpu:
                info = [tensor.cuda(rank, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]
            # else iterator is going thru a cached list with a device already assigned

            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
            pitch = pitch if pitch_guidance else None
            pitchf = pitchf if pitch_guidance else None

            # Forward pass
            use_amp = config.train.fp16_run and device.type == "cuda"
            with autocast(enabled=use_amp): # override of precision: dtype=torch.bfloat16
                model_output = net_g(
                    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
                )
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (
                    model_output
                )
                # slice of the original waveform to match a generate slice
                wave = commons.slice_segments(
                    wave,
                    ids_slice * config.data.hop_length,
                    config.train.segment_size,
                    dim=3,
                )

        # ----------   Discriminators Update   ----------
                # Zeroing gradients
                optim_d.zero_grad()

            # Run the discriminator:
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

                running_loss_disc += loss_disc.item()  # For Discriminator

        # Backward and update for discs:

            # Backward and Step for: MPD
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)


            # Clip/norm the gradients for Discriminator
            grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=1000.0)


    # Nan and Inf debugging:
        # for Discriminator
            if not math.isfinite(grad_norm_d):
                print('grad_norm_d is NaN or Inf')


            scaler.step(optim_d)

            scaler.update() # Adjust the loss scale based on the applied gradients


            # ----------   Generator Update   ----------

            # Generator backward and update
            with autocast(enabled=use_amp): # override of precision: dtype=torch.bfloat16

                # Zeroing gradients
                optim_g.zero_grad() # For Generator

            # Discriminator Loss:
                _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)

            # Loss functions for Generator:
                with autocast(enabled=False):
                    loss_fm = feature_loss(fmap_r, fmap_g) # Feature matching loss

                    loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl

                    loss_gen, _ = generator_loss(y_d_hat_g) # Generator's minimax

                    # Summed loss of generator:
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            # Backpropagation and generator optimization
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)

            # Clip/norm the gradients for Generator
            grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1000.0) 

        # Nan and Inf debugging for Generator
            if not math.isfinite(grad_norm_g):
                print('grad_norm_g is NaN or Inf')

            scaler.step(optim_g)
            scaler.update()

            global_step += 1
            pbar.update(1)
        # Accumulate losses for generator and discriminator
            running_loss_gen += loss_gen_all.item()  # For Generator

        # Logging of the averaged loss every N mini-batches
            if rank == 0 and (batch_idx + 1) % N == 0:
                avg_loss_gen = running_loss_gen / N # For Generator
                avg_loss_disc = running_loss_disc / N # For Discriminator
                writer.add_scalar('Loss/Generator_Avg', avg_loss_gen, global_step)
                writer.add_scalar('Loss/Discriminator_Avg', avg_loss_disc, global_step)
            # Resets the running loss counters
                running_loss_gen = 0.0
                running_loss_disc = 0.0

    # Logging and checkpointing
    if rank == 0:

        # used for tensorboard chart - all/mel
        mel = spec_to_mel_torch(
            spec,
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )
        # used for tensorboard chart - slice/mel_org
        y_mel = commons.slice_segments(
            mel,
            ids_slice,
            config.train.segment_size // config.data.hop_length,
            dim=3,
        )
        # used for tensorboard chart - slice/mel_gen
        with autocast(enabled=False):
            y_hat_mel = mel_spectrogram_torch(
                y_hat.float().squeeze(1),
                config.data.filter_length,
                config.data.n_mel_channels,
                config.data.sample_rate,
                config.data.hop_length,
                config.data.win_length,
                config.data.mel_fmin,
                config.data.mel_fmax,
            )
            if use_amp:
                y_hat_mel = y_hat_mel.half()

        lr = optim_g.param_groups[0]["lr"]
#        if loss_mel > 75:
#            loss_mel = 75
#        if loss_kl > 9:
#            loss_kl = 9

            # Codename;0's tweak / feature
        # Calculate the mel spectrogram similarity
        mel_spec_similarity = mel_spectrogram_similarity(y_hat_mel, y_mel)
        # Print the similarity percentage to monitor during training
        print(f'Mel Spectrogram Similarity: {mel_spec_similarity:.2f}%')

        # Logging the similarity percentage to TensorBoard
        writer.add_scalar('Metric/Mel_Spectrogram_Similarity', mel_spec_similarity, global_step)


        scalar_dict = {
            "loss/g/total": loss_gen_all,
            "loss/d/total": loss_disc,
            "learning_rate": lr,
            "grad/norm_d": grad_norm_d,
            "grad/norm_g": grad_norm_g,
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
            #"loss/g/zm": loss_zm,
        }

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }

        if epoch % save_every_epoch == 0:
            with torch.no_grad():
                if hasattr(net_g, "module"):
                    o, *_ = net_g.module.infer(*reference)
                else:
                    o, *_ = net_g.infer(*reference)
            audio_dict = {f"gen/audio_{global_step:07d}": o[0, :, :]}
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict,
                audios=audio_dict,
                audio_sample_rate=config.data.sample_rate,
            )
        else:
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict,
            )

    # Save checkpoint
    model_add = []
    done = False

    if rank == 0:

        # Extract learning rates from optimizers
        lr_g = optim_g.param_groups[0]['lr']
        lr_d = optim_d.param_groups[0]['lr']

        # Save weights every N epochs
        if epoch % save_every_epoch == 0:
            checkpoint_suffix = f"{2333333 if save_only_latest else global_step}.pth"

            # Save Generator checkpoint
            save_checkpoint(
                net_g,
                optim_g,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "G_" + checkpoint_suffix),
            )
            save_checkpoint(
                net_d,
                optim_d,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "D_" + checkpoint_suffix),
            )

            if custom_save_every_weights:
                model_add.append(
                    os.path.join(
                        experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                    )
                )


        # Check completion
        if epoch >= custom_total_epoch:
            print(
                f"Training has been successfully completed with {epoch} epoch, {global_step} steps and {round(loss_gen_all.item(), 3)} loss gen."
            )

            pid_file_path = os.path.join(experiment_dir, "config.json")
            with open(pid_file_path, "r") as pid_file:
                pid_data = json.load(pid_file)
            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)
                json.dump(pid_data, pid_file, indent=4)
            # Final model
            model_add.append(
                os.path.join(
                    experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                )
            )
            done = True

        if model_add:
            ckpt = (
                net_g.module.state_dict()
                if hasattr(net_g, "module")
                else net_g.state_dict()
            )
            for m in model_add:
                if not os.path.exists(m):
                    extract_model(
                        ckpt=ckpt,
                        sr=sample_rate,
                        pitch_guidance=pitch_guidance== True,  # converting 1/0 to True/False,
                        name=model_name,
                        model_dir=m,
                        epoch=epoch,
                        step=global_step,
                        version=version,
                        hps=hps,
                        vocoder=vocoder,
                    )
        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
        print(record)
        if done:
            os._exit(2333333)



if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
