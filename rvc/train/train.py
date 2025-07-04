import os
import re
import sys
import signal

os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"

import glob
import json
import torch
import torchaudio
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

from torch.amp import autocast

from torch.utils.data import DataLoader

from torch.nn import functional as F
import torch.nn as nn

from torch.nn.utils import clip_grad_norm_
clip_grad_norm_ = torch.nn.utils.clip_grad_norm_

import torch.distributed as dist
import torch.multiprocessing as mp

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

from losses import (
    discriminator_loss,
    generator_loss,
    wgan_discriminator_loss,
    wgan_generator_loss,
    gradient_penalty,
    feature_loss,
    feature_loss_mask,
    kl_loss,

)
from mel_processing import (
    mel_spectrogram_torch,
    spec_to_mel_torch,
    MultiScaleMelSpectrogramLoss,
)

from rvc.train.process.extract_model import extract_model
from rvc.lib.algorithm import commons
from rvc.train.custom_optimizers.ranger21 import Ranger21

import torch_optimizer

from pesq import pesq
import auraloss

# Parse command line arguments start region ===========================

model_name = sys.argv[1]
save_every_epoch = int(sys.argv[2])
total_epoch = int(sys.argv[3])
pretrainG = sys.argv[4]
pretrainD = sys.argv[5]
gpus = sys.argv[6]
batch_size = int(sys.argv[7])
sample_rate = int(sys.argv[8])
save_only_latest = strtobool(sys.argv[9])
save_every_weights = strtobool(sys.argv[10])
cache_data_in_gpu = strtobool(sys.argv[11])
use_warmup = strtobool(sys.argv[12])
warmup_duration = int(sys.argv[13])
cleanup = strtobool(sys.argv[14])
vocoder = sys.argv[15]
optimizer_choice = sys.argv[16]
use_checkpointing = strtobool(sys.argv[17])
use_tf32 = bool(strtobool(sys.argv[18]))
use_benchmark = bool(strtobool(sys.argv[19]))
use_deterministic = bool(strtobool(sys.argv[20]))
use_multiscale_mel_loss = strtobool(sys.argv[21])

double_d_update = strtobool(sys.argv[22])

if double_d_update:
    d_updates_per_step = 2
else:
    d_updates_per_step = 1
    
# Custom lr safety
use_custom_lr = strtobool(sys.argv[23])
if use_custom_lr:
    try:
        custom_lr_g = float(sys.argv[24])
        custom_lr_d = float(sys.argv[25])
    except (IndexError, ValueError):
        print("Custom LR for Generator and Discriminator is enabled, but the values aren't set properly / are invalid.")
        sys.exit(1)
else:
    custom_lr_g = None
    custom_lr_d = None

# Parse command line arguments end region ===========================

current_dir = os.getcwd()
experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "sliced_audios")

try:
    with open(config_save_path, "r") as f:
        config = json.load(f)
    config = HParams(**config)
except FileNotFoundError:
    print(
        f"Model config file not found at {config_save_path}. Did you run preprocessing and feature extraction steps?"
    )
    sys.exit(1)

config.data.training_files = os.path.join(experiment_dir, "filelist.txt")


# Globals ( do not touch these. )
global_step = 0
warmup_completed = False
from_scratch = False

# Torch backends config
torch.backends.cuda.matmul.allow_tf32 = use_tf32
torch.backends.cudnn.allow_tf32 = use_tf32
torch.backends.cudnn.benchmark = use_benchmark
torch.backends.cudnn.deterministic = use_deterministic

# Globals ( tweakable )

randomized = True
adv_weight = 1.0

log_grads_every_step = False # For debugging only

use_balancer = False # EXPERIMENTAL - Most lilely unstable.
freeze_balancer_duration = 5 #  Corresponds to a number of epochs
log_balancer = False # Logs the log_sigma for balancer

use_silence_aware_fm_loss = False  # EXPERIMENTAL - Most lilely unstable.

use_wgan_gp_loss = False  # EXPERIMENTAL - UNSTABLE
gp_weight = 1.0
precise_gp = True



import logging
logging.getLogger("torch").setLevel(logging.ERROR)

# --------------------------   Custom functions land in here   --------------------------


# Mel spectrogram similarity metric ( Predicted ∆ Real ) using L1 loss
def mel_spec_similarity(y_hat_mel, y_mel):
    # Ensure both tensors are on the same device
    device = y_hat_mel.device
    y_mel = y_mel.to(device)

    # Trim or pad tensors to the same shape (based on your preference)
    if y_hat_mel.shape != y_mel.shape:
        trimmed_shape = tuple(min(dim_a, dim_b) for dim_a, dim_b in zip(y_hat_mel.shape, y_mel.shape))
        y_hat_mel = y_hat_mel[..., :trimmed_shape[-1]]
        y_mel = y_mel[..., :trimmed_shape[-1]]
    
    # Calculate the L1 loss between the generated mel and original mel spectrograms
    loss_mel = F.l1_loss(y_hat_mel, y_mel)

    # Convert the L1 loss to a similarity score between 0 and 100
    mel_spec_similarity = 100.0 - (loss_mel * 100.0)

    # Clip the similarity percentage to ensure it stays within the desired range
    mel_spec_similarity = mel_spec_similarity.clamp(0.0, 100.0)

    return mel_spec_similarity

# Tensorboard flusher
def flush_writer(writer, rank):
    """
    Flush the TensorBoard writer if on rank 0.
    
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter
        rank (int): process rank (only rank==0 flushes)
    """
    if rank == 0 and writer is not None:
        writer.flush()

# Tensorboard flusher for grad monitoring - currently has no use.
def flush_writer_grad(writer, rank, global_step):
    """
    Flush the TensorBoard writer every 10 steps and if on rank 0.
    Dedicated for per-step gradient norm logging.
    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter
        rank (int): process rank (only rank==0 flushes)
    """
    if rank == 0 and writer is not None and global_step % 10 == 0:
        writer.flush()

# To make sure that an interrupt like Ctrl+C doesn’t flush by accident
def block_tensorboard_flush_on_exit(writer):
    def handler(signum, frame):
        print("[Warning] Training interrupted. Skipping flush to avoid partial logs.")
        try:
            writer.close()  # Close safely, no flush here.
        except:
            pass
        os._exit(1)

    signal.signal(signal.SIGINT, handler)   # for ' Ctrl+C '
    signal.signal(signal.SIGTERM, handler)  # for kill / terminate

# Scale-Invariant Signal-to-Distortion Ratio ( SI-SDR ) scoring 
def si_sdr(preds, target, eps=1e-8):
    """Scale-Invariant SDR"""
    preds = preds - preds.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    target_energy = (target ** 2).sum(dim=-1, keepdim=True)
    scaling_factor = (preds * target).sum(dim=-1, keepdim=True) / (target_energy + eps)
    projection = scaling_factor * target

    noise = preds - projection

    si_sdr_value = 10 * torch.log10((projection ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1) + eps)

    return si_sdr_value.mean()

# --------------------------   Custom functions End here   --------------------------


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


def verify_checkpoint_shapes(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_state_dict = checkpoint["model"]
    try:
        if hasattr(model, "module"):
            model_state_dict = model.module.load_state_dict(checkpoint_state_dict)
        else:
            model_state_dict = model.load_state_dict(checkpoint_state_dict)
    except RuntimeError:
        print(
            "The parameters of the pretrain model such as the sample rate or architecture do not match the selected model."
        )
        sys.exit(1)
    else:
        del checkpoint
        del checkpoint_state_dict
        del model_state_dict


def main():
    """
    Main function to start the training process.
    """
    global gpus

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
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
        gpus = [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus) 
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        gpus = [0]
        n_gpus = 1
    else:
        device = torch.device("cpu")
        gpus = [0]
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
            for rank, device_id in enumerate(gpus):
                subproc = mp.Process(
                    target=run,
                    args=(
                        rank,
                        n_gpus,
                        experiment_dir,
                        pretrainG,
                        pretrainD,
                        total_epoch,
                        save_every_weights,
                        config,
                        device,
                        device_id,
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
    custom_total_epoch,
    custom_save_every_weights,
    config,
    device,
    device_id,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrainG (str): Path to the pre-trained generator model.
        pretrainD (str): Path to the pre-trained discriminator model.
        custom_total_epoch (int): The total number of epochs for training.
        custom_save_every_weights (int): The interval (in epochs) at which to save model weights.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step, warmup_completed, optimizer_choice, from_scratch


    if 'warmup_completed' not in globals():
        warmup_completed = False

    # Warmup init msg:
    if rank == 0 and use_warmup:
        print(f"    ██████  Warmup Enabled for {warmup_duration} epochs. ██████")

    # Precision init msg:
    if not config.train.bf16_run:
        if torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32:
            print("    ██████  PRECISION: TF32                               ██████")
        else:
            print("    ██████  PRECISION: FP32                               ██████")
    else:
        print("    ██████  PRECISION: BrainFloat16 AMP                   ██████")

    # backends.cudnn checks:
        # For benchmark:
    if torch.backends.cudnn.benchmark:
        print("    ██████  cudnn.benchmark: True                         ██████")
    else:
        print("    ██████  cudnn.benchmark: False                        ██████")
        # For deterministic:
    if torch.backends.cudnn.deterministic:
        print("    ██████  cudnn.deterministic: True                     ██████")
    else:
        print("    ██████  cudnn.deterministic: False                    ██████")

    # optimizer checks:
        # For Ranger21:
    if optimizer_choice == "Ranger21":
        print("    ██████  Optimizer used: Ranger21                      ██████")
        # For RAdam:
    elif optimizer_choice == "RAdam":
        print("    ██████  Optimizer used: RAdam                         ██████")
        # For AdamW:
    elif optimizer_choice == "AdamW":
        print("    ██████  Optimizer used: AdamW                         ██████")

    # Training strategy checks:
    if d_updates_per_step == 2:
        print("    ██████  Double-update for Discriminator: Yes          ██████")
        print(f"    ██████  Amount of D updates per step: {d_updates_per_step}              ██████")

    # Uncertainty loss balancer checks:
    if use_balancer:
        print("    ██████  Uncertainty loss balancer: Yes                ██████")
        if freeze_balancer_duration > 0:
            print(f"    ██████  balancer frozen until epoch: {freeze_balancer_duration }               ██████")
    else:
        print("    ██████  Uncertainty loss balancer: No                 ██████")

    # Losses checks:
    if use_silence_aware_fm_loss:
        print("    ██████  Using silence aware fm loss                   ██████")
    if use_wgan_gp_loss:
        if not precise_gp:
            print("    ██████  Using WGAN-GP loss for G and D                ██████")
        else:
            print("    ██████  Using WGAN-GP loss for G and D ( precise )    ██████")



    if rank == 0:
        writer_eval = SummaryWriter(
            log_dir=os.path.join(experiment_dir, "eval"),
            flush_secs=86400 # Periodic background flush's timer workarouand.
        )
        block_tensorboard_flush_on_exit(writer_eval)
    else:
        writer_eval = None

    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    # Create datasets and dataloaders
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid,
    )

    full_dataset = TextAudioLoaderMultiNSFsid(config.data)
    collate_fn = TextAudioCollateMultiNSFsid()

    # Split dataset
    train_len = int(0.90 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(config.train.seed),
    )

    train_dataset.lengths = [full_dataset.lengths[i] for i in train_dataset.indices]
    val_dataset.lengths = [full_dataset.lengths[i] for i in val_dataset.indices]

    # Distributed sampler and loader for train set
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

    # Distributed sampler and loader for eval set
    val_sampler = DistributedBucketSampler(
        val_dataset,
        batch_size * n_gpus,
        [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=False, # Off for validation purposes.
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=True,
    )

    # train_loader safety check
    if len(train_loader) < 3:
        print(
            "Not enough data present in the training set. Perhaps you didn't slice the audio files? ( Preprocessing step )"
        )
        os._exit(2333333)

    # Initialize models and optimizers
    from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
    from rvc.lib.algorithm.synthesizers import Synthesizer

    net_g = Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0 = True,
        sr = sample_rate,
        vocoder = vocoder,
        checkpointing = use_checkpointing,
        randomized = randomized,
    )

    net_d = MultiPeriodDiscriminator(
        config.model.use_spectral_norm, use_checkpointing=use_checkpointing
    )

    if torch.cuda.is_available():
        net_g = net_g.cuda(device_id)
        net_d = net_d.cuda(device_id)
    else:
        net_g = net_g.to(device)
        net_d = net_d.to(device)


    class LossBalancer(nn.Module):
        def __init__(self, freeze_until_epoch=3, freeze_list=None):
            super().__init__()

            self.freeze_until_epoch = freeze_until_epoch
            self.freeze_list = freeze_list or []

            # Initialize raw log_sigma values
            # PS. use this formula to convert * weight to log sigma:  -0.5 * log(2 * W)
            self.log_sigma_adv = nn.Parameter(torch.tensor(-0.3466))   # ≈ 1.0
            self.log_sigma_mel = nn.Parameter(torch.tensor(-2.2499))   # ≈ 45.0
            self.log_sigma_fm  = nn.Parameter(torch.tensor(-0.6931))   # ≈ 2.0
            self.log_sigma_kl  = nn.Parameter(torch.tensor(-0.3466))   # ≈ 1.0

            # Init values to reuse during warmup
            self.fixed_log_sigmas = {
                "adv": -0.3466,
                "mel": -2.2499,
                "fm": -0.6931,
                "kl": -0.3466
            }

        def forward(self, loss_adv, loss_mel, loss_fm, loss_kl, epoch):
            log_sigmas = {}

            for name in ["adv", "mel", "fm", "kl"]:
                param = getattr(self, f"log_sigma_{name}")
                if epoch < self.freeze_until_epoch and name in self.freeze_list:
                    log_sigmas[name] = torch.tensor(self.fixed_log_sigmas[name], device=param.device)
                else:
                    log_sigmas[name] = param

            weighted_adv = (1.0 / (2 * torch.exp(log_sigmas["adv"])**2)) * loss_adv + log_sigmas["adv"]
            weighted_mel = (1.0 / (2 * torch.exp(log_sigmas["mel"])**2)) * loss_mel + log_sigmas["mel"]
            weighted_fm  = (1.0 / (2 * torch.exp(log_sigmas["fm"])**2))  * loss_fm  + log_sigmas["fm"]
            weighted_kl  = (1.0 / (2 * torch.exp(log_sigmas["kl"])**2))  * loss_kl  + log_sigmas["kl"]

            return weighted_adv + weighted_mel + weighted_fm + weighted_kl

    loss_balancer = LossBalancer(
        freeze_until_epoch=freeze_balancer_duration,
        freeze_list=["adv", "mel", "fm", "kl"]
    ).to(device)


    # Needed for the uncertainty weighted loss balancer ( Doesn't matter which optim's in use. )
    gen_params = list(net_g.parameters())
    if use_balancer:
        gen_params += list(loss_balancer.parameters())

    # Base / Common kwargs for gen and disc
    common_args_g = dict(
        lr=custom_lr_g if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0,
    )
    common_args_d = dict(
        lr=custom_lr_d if use_custom_lr else config.train.learning_rate,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=0,
    )

    if optimizer_choice == "Ranger21":
        extra_args = dict(
            num_epochs=custom_total_epoch,
            num_batches_per_epoch=len(train_loader),
            use_madgrad=False,
            use_warmup=False,
            warmdown_active=False,
            use_cheb=False,
            lookahead_active=True,
            normloss_active=False,
            normloss_factor=1e-4,
            softplus=False,
            use_adaptive_gradient_clipping=True,
            agc_clipping_value=0.01,
            agc_eps=1e-3,
            using_gc=True,
            gc_conv_only=True,
            using_normgc=False,
        )
        optim_g = Ranger21(gen_params, **common_args_g, **extra_args)
        optim_d = Ranger21(net_d.parameters(), **common_args_d, **extra_args)

    elif optimizer_choice == "RAdam":
        optim_g = torch_optimizer.RAdam(gen_params, **common_args_g)
        optim_d = torch_optimizer.RAdam(net_d.parameters(), **common_args_d)

    elif optimizer_choice == "AdamW":
        optim_g = torch.optim.AdamW(gen_params, **common_args_g)
        optim_d = torch.optim.AdamW(net_d.parameters(), **common_args_d)

    '''
    - Template for adding custom optims:

    elif optimizer_choice =="new_optim":
        extra_args = dict(
            first = 0.123,
            second = 123,
            third = 0.123123,
        )
        optim_g = new_optim(gen_params, **common_args_g, **extra_args)
        optim_d = new_optim(net_d.parameters(), **common_args_d, **extra_args)
    '''

    if use_multiscale_mel_loss:
        fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=sample_rate)
        print("    ██████  Using Multi-Scale Mel loss function           ██████")
    else:
        fn_mel_loss = torch.nn.L1Loss()
        print("    ██████  Using Single-Scale (L1) Mel loss function     ██████")


    # Wrap models with DDP for multi-gpu processing
    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id]) # find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[device_id]) # find_unused_parameters=True)

    # Load checkpoint if available
    try:
        print("    ██████  Starting the training ...                     ██████")
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
                verify_checkpoint_shapes(pretrainG, net_g)
                print(f"Loaded pretrained (G) '{pretrainG}'")
            if hasattr(net_g, "module"):
                net_g.module.load_state_dict(
                    torch.load(pretrainG, map_location="cpu", weights_only=True)["model"]
                )
            else:
                net_g.load_state_dict(
                    torch.load(pretrainG, map_location="cpu", weights_only=True)["model"]
                )

    # Loading the pretrained Discriminator model
        if pretrainD != "" and pretrainD != "None":
            if rank == 0:
                print(f"Loaded pretrained (D) '{pretrainD}'")
            if hasattr(net_d, "module"):
                net_d.module.load_state_dict(
                    torch.load(pretrainD, map_location="cpu", weights_only=True)["model"]
                )
            else:
                net_d.load_state_dict(
                    torch.load(pretrainD, map_location="cpu", weights_only=True)["model"]
                )

    # Check if the training is ' from scratch ' and set appropriate flag
    if (pretrainG in ["", "None"]) and (pretrainD in ["", "None"]):
        from_scratch = True
        if rank == 0:
            print("    ██████  No pretrained loaded: TRAINING FROM SCRATCH.  ██████")

    # Initialize the warmup scheduler only if `use_warmup` is True
    if use_warmup:
        # Warmup for: Generator
        warmup_scheduler_g = torch.optim.lr_scheduler.LambdaLR(
            optim_g,
            lr_lambda=lambda epoch: (epoch + 1) / warmup_duration if epoch < warmup_duration else 1.0
        )
        # Warmup for: MPD
        warmup_scheduler_d = torch.optim.lr_scheduler.LambdaLR(
            optim_d,
            lr_lambda=lambda epoch: (epoch + 1) / warmup_duration if epoch < warmup_duration else 1.0
        )

    # Ensure initial_lr is set when use_warmup is False
    if not use_warmup:
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
#    decay_scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=50, eta_min=3e-5)
    decay_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=0.999875, last_epoch=epoch_str - 1  #  ( stock ) 0.999875      ( finetuning ) 0.995   <=>  ( 50% slower ) 0.9975     ( 30% slower ) 0.9965
    )

        # For: Discriminator
#    decay_scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=50, eta_min=3e-5)
    decay_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=0.999875, last_epoch=epoch_str - 1  #  ( stock ) 0.999875       ( finetuning ) 0.995   <=>  ( 50% slower ) 0.9975     ( 30% slower ) 0.9965
    )

    # Reference sample fetching mechanism:
        # Feel free to customize the path or namings ( make sure to change em in ' if ' block too. )
    reference_path = os.path.join("logs", "reference")
    use_custom_ref = all([
        os.path.isfile(os.path.join(reference_path, "ref_feats.npy")),    # Features
        os.path.isfile(os.path.join(reference_path, "ref_f0c.npy")),      # Pitch - Coarse
        os.path.isfile(os.path.join(reference_path, "ref_f0f.npy")),      # Pitch - Float
    ])

    cache = []

    if use_custom_ref:
        print("Using custom reference input from 'logs\\reference\\'")

        # Load and process
        phone = np.load(os.path.join(reference_path, "ref_feats.npy"))
        phone = np.repeat(phone, 2, axis=0)  # Match pitch frame rate
        pitch = np.load(os.path.join(reference_path, "ref_f0c.npy"))
        pitchf = np.load(os.path.join(reference_path, "ref_f0f.npy"))

        # Find minimum length
        min_len = min(len(phone), len(pitch), len(pitchf))
        
        # Trim all to same length
        phone = phone[:min_len]
        pitch = pitch[:min_len]
        pitchf = pitchf[:min_len]

        # Convert to tensors
        phone = torch.FloatTensor(phone).unsqueeze(0).to(device)
        phone_lengths = torch.LongTensor([phone.shape[1]]).to(device)
        pitch = torch.LongTensor(pitch).unsqueeze(0).to(device)
        pitchf = torch.FloatTensor(pitchf).unsqueeze(0).to(device)

        sid = torch.LongTensor([0]).to(device)  # default speaker
        reference = (phone, phone_lengths, pitch, pitchf, sid)

    else:
        print("No custom reference found, perhaps a mistake in filename?")
        print("[FALLBACK] Using the first batch from train_loader.")
        info = next(iter(train_loader))
        phone, phone_lengths, pitch, pitchf, _, _, _, _, sid = info
        reference = (
            phone.to(device, non_blocking=True),
            phone_lengths.to(device, non_blocking=True),
            pitch.to(device, non_blocking=True),
            pitchf.to(device, non_blocking=True),
            sid.to(device, non_blocking=True),
        )

    for epoch in range(epoch_str, total_epoch + 1):
        training_loop(
            rank,
            epoch,
            config,
            [net_g, net_d],
            [optim_g, optim_d],
            [train_loader, val_loader],
            [writer_eval],
            cache,
            custom_save_every_weights,
            custom_total_epoch,
            device,
            device_id,
            reference,
            fn_mel_loss,
            n_gpus,
            loss_balancer,
        )

        if use_warmup and epoch <= warmup_duration:
            # Starts the warmup phase if warmup_duration =/= warmup_duration
            warmup_scheduler_g.step()
            warmup_scheduler_d.step()

            # Logging of finished warmup
            if epoch == warmup_duration:
                warmup_completed = True
                print(f"    ██████  Warmup completed at pochs: {warmup_duration}  ██████")
                # Gen:
                print(f"    ██████  LR G: {optim_g.param_groups[0]['lr']}         ██████")
                # Discs:
                print(f"    ██████  LR D: {optim_d.param_groups[0]['lr']}         ██████")
                # Decay gamma:
                print(f"    ██████  Starting the exponential lr decay with gamma of {config.train.lr_decay}  ██████")
 
        # Once the warmup phase is completed, uses exponential lr decay
        if not use_warmup or warmup_completed:
            decay_scheduler_g.step()
            decay_scheduler_d.step()


def training_loop(
    rank,
    epoch,
    hps,
    nets,
    optims,
    loaders,
    writers,
    cache,
    custom_save_every_weights,
    custom_total_epoch,
    device,
    device_id,
    reference,
    fn_mel_loss,
    n_gpus,
    loss_balancer,
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        hps (Namespace): Hyperparameters.
        nets (list): List of models [net_g, net_d].
        optims (list): List of optimizers [optim_g, optim_d].
        loaders (list): List of dataloaders [train_loader, eval_loader].
        writers (list): List of TensorBoard writers [writer_eval].
        cache (list): List to cache data in GPU memory.
        use_cpu (bool): Whether to use CPU for training.
    """
    global global_step, warmup_completed

    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader = loaders[0] if loaders is not None else None
    val_loader = loaders[1] if loaders is not None and len(loaders) > 1 else None

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
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
                cache.append((batch_idx, info))
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()

    if not from_scratch:
        # Tensors init for averaged losses:
        tensor_count = 6
        if use_wgan_gp_loss:
            tensor_count = 8

        epoch_loss_tensor = torch.zeros(tensor_count, device=device)
        multi_epoch_loss_tensor = torch.zeros(tensor_count, device=device)
        num_batches_in_epoch = 0

    avg_50_cache = {
        "grad_norm_d_raw_50": deque(maxlen=50),
        "grad_norm_g_raw_50": deque(maxlen=50),
        "grad_norm_d_clipped_50": deque(maxlen=50),
        "grad_norm_g_clipped_50": deque(maxlen=50),
        "discriminator_adv_50": deque(maxlen=50),
        "generator_adv_50": deque(maxlen=50),
        "generator_total_50": deque(maxlen=50),
        "fm_50": deque(maxlen=50),
        "mel_50": deque(maxlen=50),
        "kl_50": deque(maxlen=50),
    }
    if use_wgan_gp_loss:
        avg_50_cache["wasserstein_distance_50"] = deque(maxlen=50)
        avg_50_cache["gradient_penalty_50"] = deque(maxlen=50)

    if log_grads_every_step:
        # buffering for logging of grads:
        grad_step_log_buffer = {
        "grad/norm_d_raw_step": [],
        "grad/norm_g_raw_step": [],
        "grad/norm_d_raw_step_clipped": [],
        "grad/norm_g_raw_step_clipped": [],
        }

    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:

            global_step += 1

            if not from_scratch:
                num_batches_in_epoch += 1

            if device.type == "cuda" and not cache_data_in_gpu:
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]
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

            use_amp = config.train.bf16_run and device.type == "cuda"

            # Generator forward pass:
            with autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
                model_output = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
                # Unpacking:

                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (model_output)

                # Slice the original waveform to match the generated slice:
                if randomized:
                    wave = commons.slice_segments(
                        wave,
                        ids_slice * config.data.hop_length,
                        config.train.segment_size,
                        dim=3,
                    )

            # Discriminator forward pass:
            for _ in range(d_updates_per_step):  # default is 1 update per step
                with autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())

                # Compute discriminator loss:
                if not use_wgan_gp_loss:
                    loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
                else:
                    loss_disc = wgan_discriminator_loss(y_d_hat_r, y_d_hat_g)

                    d_real = torch.mean(torch.stack([dr.mean().detach() for dr in y_d_hat_r]))
                    d_fake = torch.mean(torch.stack([dg.mean().detach() for dg in y_d_hat_g]))
                    wasserstein_distance = d_real - d_fake

                    gp = gradient_penalty(net_d, wave, y_hat.detach(), device, average_all_outputs=precise_gp)
                    loss_disc = loss_disc + gp_weight * gp

                # Discriminator backward and update:
                optim_d.zero_grad()
                loss_disc.backward()
                # 1. Retrieve raw grads norm
                grad_norm_d_raw = commons.get_total_norm([p.grad for p in net_d.parameters() if p.grad is not None], norm_type=2.0, error_if_nonfinite=True)
                if log_grads_every_step:
                    grad_log_buffer["grad/norm_d_raw_step"].append((global_step, grad_norm_d_raw))
                # 2. Grads norm clip
                grad_norm_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=999999) # 1000 / 999999
                # 3. Retrieve the clipped grads
                grad_norm_d_clipped = commons.get_total_norm([p.grad for p in net_d.parameters() if p.grad is not None], norm_type=2.0, error_if_nonfinite=True)
                if log_grads_every_step:
                    grad_log_buffer["grad/norm_d_raw_step_clipped"].append((global_step, grad_norm_d_clipped))
                # 4. Optimization step
                optim_d.step()


            # Run discriminator on generated output
            with autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
                _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)


            # Compute generator losses:
            if use_multiscale_mel_loss:
                if not use_balancer:
                    loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
                else:
                    loss_mel = fn_mel_loss(wave, y_hat)
            else:
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
                mel = spec_to_mel_torch(
                    spec,
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sample_rate,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                y_mel = commons.slice_segments(mel, ids_slice, config.train.segment_size // config.data.hop_length, dim=3)
                if not use_balancer:
                    loss_mel = fn_mel_loss(y_mel, y_hat_mel) * config.train.c_mel
                else:
                    loss_mel = fn_mel_loss(y_mel, y_hat_mel)

            # Feature Matching loss
            if not use_balancer:
                loss_fm = feature_loss(fmap_r, fmap_g) * 2.0 # Previously was * 2 internally, moved outside for the sake of balancer.
            else:
                if not use_silence_aware_fm_loss:
                    loss_fm = feature_loss(fmap_r, fmap_g)
                else:
                    with torch.no_grad():
                        energy = wave.abs().mean(dim=[1, 2])
                        silence_mask = (energy / (energy.max() + 1e-6)).clamp(min=0.05, max=1.0)  # min vals: 0 - full silence masking, 0.05 - safe default, 0.10 for fuzzy / or breathy silence.
                    loss_fm = feature_loss_mask(fmap_r, fmap_g, silence_mask=silence_mask)

            # Generator loss 
            if not use_wgan_gp_loss:
                loss_gen = generator_loss(y_d_hat_g)
            else:
                loss_gen = wgan_generator_loss(y_d_hat_g)

            # KL ( Kullback–Leibler divergence ) loss
            if not use_balancer:
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
            else:
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)

            # Total generator loss
            if not use_balancer:
                loss_gen_all = ( loss_gen * adv_weight ) + loss_fm + loss_mel + loss_kl
            else:
                loss_gen_all = loss_balancer(loss_gen, loss_mel, loss_fm, loss_kl, epoch) 

            # Logging of the balancer and optionally silence
            if log_balancer and use_balancer and rank == 0:
                if use_silence_aware_fm_loss:
                    silence_ratio = 1.0 - silence_mask.mean().item()
                    writer.add_scalar("Debug/silence_ratio", silence_ratio, global_step)
                loss_map = {"adv": loss_gen, "mel": loss_mel, "fm":  loss_fm, "kl":  loss_kl}
                for name in ["adv", "mel", "fm", "kl"]:
                    log_sigma = getattr(loss_balancer, f"log_sigma_{name}")
                    sigma = torch.exp(log_sigma)
                    weight = 1 / (2 * sigma ** 2)
                    contribution = weight * loss_map[name].detach()
                    writer.add_scalar(f"Balancer/log_sigma_{name}", log_sigma.item(), global_step)
                    writer.add_scalar(f"Balancer/weight_{name}", weight.item(), global_step)
                    writer.add_scalar(f"Balancer/contribution_{name}", contribution.item(), global_step)


            # Generator backward and update:
            optim_g.zero_grad()
            loss_gen_all.backward()
            # 1. Retrieve raw grads norm
            grad_norm_g_raw = commons.get_total_norm([p.grad for p in net_g.parameters() if p.grad is not None], norm_type=2.0, error_if_nonfinite=True)
            if log_grads_every_step:
                grad_log_buffer["grad/norm_g_raw_step"].append((global_step, grad_norm_g_raw))
            # 2. Grads norm clip
            grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=999999) # 1000 / 999999
            # 3. Retrieve the clipped grads
            grad_norm_g_clipped = commons.get_total_norm([p.grad for p in net_g.parameters() if p.grad is not None], norm_type=2.0, error_if_nonfinite=True)
            if log_grads_every_step:
                grad_log_buffer["grad/norm_g_raw_step_clipped"].append((global_step, grad_norm_g_clipped))
            # 4. Optimization step
            optim_g.step()

            if not from_scratch:
                # Loss accumulation In the epoch_loss_tensor
                epoch_loss_tensor[0].add_(loss_disc.detach())
                epoch_loss_tensor[1].add_(loss_gen.detach())
                epoch_loss_tensor[2].add_(loss_gen_all.detach())
                epoch_loss_tensor[3].add_(loss_fm.detach())
                epoch_loss_tensor[4].add_(loss_mel.detach())
                epoch_loss_tensor[5].add_(loss_kl.detach())
                if use_wgan_gp_loss:
                    epoch_loss_tensor[6].add_(wasserstein_distance.detach())
                    epoch_loss_tensor[7].add_(gp.detach())

            # queue for rolling losses / grads over 50 steps
            # Grads:
            avg_50_cache["grad_norm_d_raw_50"].append(grad_norm_d_raw)
            avg_50_cache["grad_norm_g_raw_50"].append(grad_norm_g_raw)
            avg_50_cache["grad_norm_d_clipped_50"].append(grad_norm_d_clipped)
            avg_50_cache["grad_norm_g_clipped_50"].append(grad_norm_g_clipped)
            # Losses:
            avg_50_cache["discriminator_adv_50"].append(loss_disc.detach())
            avg_50_cache["generator_adv_50"].append(loss_gen.detach())
            avg_50_cache["generator_total_50"].append(loss_gen_all.detach())
            avg_50_cache["fm_50"].append(loss_fm.detach())
            avg_50_cache["mel_50"].append(loss_mel.detach())
            avg_50_cache["kl_50"].append(loss_kl.detach())
            if use_wgan_gp_loss:
                avg_50_cache["wasserstein_distance_50"].append(wasserstein_distance.detach())
                avg_50_cache["gradient_penalty_50"].append(gp.detach())

            if rank == 0 and global_step % 50 == 0:
                scalar_dict_50 = {}
                # Learning rate retrieval for avg-50 variation:
                if from_scratch:
                    lr_d = optim_d.param_groups[0]["lr"]
                    lr_g = optim_g.param_groups[0]["lr"]
                    scalar_dict_50.update({
                    "learning_rate/lr_d": lr_d,
                    "learning_rate/lr_g": lr_g,
                    })
                # logging rolling averages
                scalar_dict_50.update({
                    # Grads:
                    "grad_avg_50/norm_d_raw_50": sum(avg_50_cache["grad_norm_d_raw_50"])
                    / len(avg_50_cache["grad_norm_d_raw_50"]),
                    "grad_avg_50/norm_g_raw_50": sum(avg_50_cache["grad_norm_g_raw_50"])
                    / len(avg_50_cache["grad_norm_g_raw_50"]),
                    "grad_avg_50/norm_d_clipped_50": sum(avg_50_cache["grad_norm_d_clipped_50"])
                    / len(avg_50_cache["grad_norm_d_clipped_50"]),
                    "grad_avg_50/norm_g_clipped_50": sum(avg_50_cache["grad_norm_g_clipped_50"])
                    / len(avg_50_cache["grad_norm_g_clipped_50"]),
                    # Losses:
                    "loss_avg_50/discriminator_adv_50": torch.mean(
                        torch.stack(list(avg_50_cache["discriminator_adv_50"]))),
                    "loss_avg_50/generator_adv_50": torch.mean(
                        torch.stack(list(avg_50_cache["generator_adv_50"]))),
                    "loss_avg_50/generator_total_50": torch.mean(
                        torch.stack(list(avg_50_cache["generator_total_50"]))),
                    "loss_avg_50/fm_50": torch.mean(
                        torch.stack(list(avg_50_cache["fm_50"]))),
                    "loss_avg_50/mel_50": torch.mean(
                        torch.stack(list(avg_50_cache["mel_50"]))),
                    "loss_avg_50/kl_50": torch.mean(
                        torch.stack(list(avg_50_cache["kl_50"]))),
                })
                if use_wgan_gp_loss:
                    scalar_dict_50["WGAN_Metrics/wasserstein_distance_50"] = torch.mean(
                        torch.stack(list(avg_50_cache["wasserstein_distance_50"]))
                    )
                    scalar_dict_50["WGAN_Metrics/gradient_penalty_50"] = torch.mean(
                        torch.stack(list(avg_50_cache["gradient_penalty_50"]))
                    )
                summarize(writer=writer, global_step=global_step, scalars=scalar_dict_50)
                flush_writer(writer, rank)

            pbar.update(1)
        # end of batch train
    # end of tqdm

    if n_gpus > 1 and device.type == 'cuda':
        dist.barrier()

    with torch.no_grad():
        torch.cuda.empty_cache()

    # Logging and checkpointing
    if rank == 0:
        # Used for tensorboard chart - all/mel
        mel = spec_to_mel_torch(
            spec,
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )

        # Used for tensorboard chart - slice/mel_org
        if randomized:
            y_mel = commons.slice_segments(
                mel,
                ids_slice,
                config.train.segment_size // config.data.hop_length,
                dim=3,
            )
        else:
            y_mel = mel

        # used for tensorboard chart - slice/mel_gen
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
        # Mel similarity metric:
        mel_similarity = mel_spec_similarity(y_hat_mel, y_mel)
        print(f'Mel Spectrogram Similarity: {mel_similarity:.2f}%')
        writer.add_scalar('Metric/Mel_Spectrogram_Similarity', mel_similarity, global_step)

        # Learning rate retrieval for avg-epoch variation:
        lr_d = optim_d.param_groups[0]["lr"]
        lr_g = optim_g.param_groups[0]["lr"]

        # Calculate the avg epoch loss:
        if global_step % len(train_loader) == 0 and not from_scratch: # At each epoch completion
            avg_epoch_loss = epoch_loss_tensor / num_batches_in_epoch

            scalar_dict_avg = {
            "loss_avg/discriminator_adv": avg_epoch_loss[0],
            "loss_avg/generator_adv": avg_epoch_loss[1],
            "loss_avg/generator_total": avg_epoch_loss[2],
            "loss_avg/fm": avg_epoch_loss[3],
            "loss_avg/mel": avg_epoch_loss[4],
            "loss_avg/kl": avg_epoch_loss[5],
            "learning_rate/lr_d": lr_d,
            "learning_rate/lr_g": lr_g,
            }

            if use_wgan_gp_loss:
                scalar_dict_avg["wgan_metrics_avg/wasserstein_distance"] = avg_epoch_loss[6]
                scalar_dict_avg["wgan_metrics_avg/gradient_penalty"] = avg_epoch_loss[7]

            summarize(writer=writer, global_step=global_step, scalars=scalar_dict_avg)
            flush_writer(writer, rank)
            num_batches_in_epoch = 0
            epoch_loss_tensor.zero_()

        # Logging of gradients using " log by step " approach:
        if log_grads_every_step:
            if global_step % len(train_loader) == 0:
                for tag, values in grad_step_log_buffer.items():
                    for step, val in values:
                        writer.add_scalar(tag, val, step)
                grad_step_log_buffer = {k: [] for k in grad_step_log_buffer}

        image_dict = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].detach().cpu().to(torch.float32).numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].detach().cpu().to(torch.float32).numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].detach().cpu().to(torch.float32).numpy()),
        }

        # At each epoch save point:
        if epoch % save_every_epoch == 0:
            # Running validation
            validation_loop(
                net_g.module if hasattr(net_g, "module") else net_g,
                val_loader,
                device,
                hps,
                writer,
                global_step,
            )
            # Inferencing on reference sample
            net_g.eval()
            with torch.no_grad():
                if hasattr(net_g, "module"):
                    o, *_ = net_g.module.infer(*reference)
                else:
                    o, *_ = net_g.infer(*reference)
            net_g.train()
            audio_dict = {f"gen/audio_{global_step:07d}": o[0, :, :]} # Eval-infer samples
            # Logging
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                audios=audio_dict,
                audio_sample_rate=config.data.sample_rate,
            )
            flush_writer(writer, rank)
        else:
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
            )
            flush_writer(writer, rank)

    # Save checkpoint
    model_add = []
    done = False

    if rank == 0:
        # Print training progress
        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
        print(record)

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
            # Save Discriminator checkpoint
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
                        name=model_name,
                        model_path=m,
                        epoch=epoch,
                        step=global_step,
                        hps=hps,
                        vocoder=vocoder,
                    )
        if done:
            # Clean-up process IDs from config.json
            pid_file_path = os.path.join(experiment_dir, "config.json")
            with open(pid_file_path, "r") as pid_file:
                try:
                    pid_data = json.load(pid_file)
                except json.JSONDecodeError:
                    pid_data = {}

            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)
                json.dump(pid_data, pid_file, indent=4)

            if rank == 0:
                writer.flush()
                writer.close()

            os._exit(2333333)

        with torch.no_grad():
            torch.cuda.empty_cache()


def validation_loop(net_g, val_loader, device, hps, writer, global_step):
    net_g.eval()
    torch.cuda.empty_cache()

    total_mel_error = 0.0
    total_mrstft_loss = 0.0
    total_pesq = 0.0
    valid_pesq_count = 0
    total_si_sdr = 0.0
    count = 0

    mrstft = auraloss.freq.MultiResolutionSTFTLoss(device=device)
    resample_to_16k = torchaudio.transforms.Resample(orig_freq=hps.data.sample_rate, new_freq=16000).to(device)

    hop_length = hps.data.hop_length
    sample_rate = hps.data.sample_rate

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid = [
                t.to(device) for t in batch
            ]

            # Infer
            y_hat, x_mask, _ = net_g.infer(phone, phone_lengths, pitch, pitchf, sid)

            wave_len = wave.shape[-1]

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                sample_rate,
                hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                sample_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            # Mel loss:
            y_hat_mel_len = y_hat_mel.shape[-1]
            mel_len = mel.shape[-1]

            min_t = min(y_hat_mel_len, mel_len)

            mel_loss = F.l1_loss(y_hat_mel[..., :min_t], mel[..., :min_t])
            total_mel_error += mel_loss.item()

            # STFT loss:
            y_hat_len = y_hat.shape[-1]

            min_samples = min_t * hop_length
            min_samples = min(min_samples, wave_len, y_hat_len)

            stft_loss = mrstft(y_hat[..., :min_samples], wave[..., :min_samples])
            total_mrstft_loss += stft_loss.item()

            # si_sdr:
            si_sdr_score = si_sdr(y_hat.squeeze(1), wave.squeeze(1))
            total_si_sdr += si_sdr_score.item()

            # PESQ:
            try:
                y_16k_batch = resample_to_16k(wave).cpu().numpy()          # (B, T)
                y_hat_16k_batch = resample_to_16k(y_hat.squeeze(1)).cpu().numpy()  # (B, T)

                for i in range(y_16k_batch.shape[0]):
                    y_16k_f = np.squeeze(y_16k_batch[i]).astype(np.float32)
                    y_hat_16k_f = np.squeeze(y_hat_16k_batch[i]).astype(np.float32)

                    try:
                        pesq_score = pesq(16000, y_16k_f, y_hat_16k_f, mode="wb")
                        total_pesq += pesq_score
                        valid_pesq_count += 1
                    except Exception as e:
                        print(f"[PESQ skipped] {e}")

            except Exception as e:
                print(f"[PESQ skipped outer] {e}")

            count += 1

    avg_mel = total_mel_error / count
    avg_mrstft = total_mrstft_loss / count
    avg_pesq = total_pesq / max(valid_pesq_count, 1)
    avg_si_sdr = total_si_sdr / count

    if writer is not None:
        writer.add_scalar("validation/loss/mel_l1", avg_mel, global_step)
        writer.add_scalar("validation/loss/mrstft", avg_mrstft, global_step)
        writer.add_scalar("validation/score/pesq", avg_pesq, global_step)
        writer.add_scalar("validation/score/si_sdr", avg_si_sdr, global_step)

    net_g.train()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
