import torch
import torch.nn as nn
import numpy as np

from models.s4 import S4Block

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


################################################################################
# Fisher Information Matrix Regularized Training
# "Overcoming Catastrophic Forgetting in Neural Networks" (Kirkpatrick et al, 2017)

class FIMRegularizedTraining:
    def __init__(self, model, alpha=1.0):
        self.model = model
        self.alpha = alpha  # Regularization coefficient
        self.original_params = {}  # To store the original parameters
        self.fims = {}  # To store the normalized FIMs

        self.store_model()

    def store_model(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.original_params[name] = param.data.clone().detach()
                fim = param.grad ** 2
                fim_norm = torch.norm(fim, p=1)
                self.fims[name] = fim / fim_norm if fim_norm > 0 else fim

    def accumulate_model(self, beta=0.5):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.original_params[name] = param.data.clone().detach()

                fim = param.grad ** 2
                fim_norm = torch.norm(fim, p=1)
                fim = fim / fim_norm if fim_norm > 0 else fim

                # EWMA and re-normalize
                fim = self.fims[name] * beta + fim * (1 - beta)
                fim_norm = torch.norm(fim, p=1)
                fim = fim / fim_norm if fim_norm > 0 else fim

                self.fims[name] = fim

    def loss(self):
        fim_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fims:  # Ensure the parameter was stored and has a corresponding FIM
                fim_contribution = self.fims[name] * (param - self.original_params[name]).pow(2)
                fim_loss += fim_contribution.sum()
        return self.alpha * fim_loss


################################################################################
# Model

import torch.nn.functional as F

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x):
        return self.norm(self.fn(x) + x)

class S4Layer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = S4Block(dim)

    def forward(self, x):
        x = x.transpose(-1, -2) # [B, L, D] -> [B, D, L]
        y, _ = self.net(x)
        y = y.transpose(-1, -2) # [B, D, L] -> [B, L, D]
        return y

class AudioS4(nn.Module):
    def __init__(self, dim, num_layers=3):
        super().__init__()

        d_hidden = dim * 2
        self.proj_in = nn.Linear(dim, d_hidden)
        self.proj_out = nn.Linear(d_hidden, dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = PreNormResidual(d_hidden, S4Layer(d_hidden))
            self.layers.append(layer)

    def forward(self, x):
        x = self.proj_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.proj_out(x)
        return x


################################################################################
# Dataset

from torch.utils.data import Dataset, DataLoader, Subset

import torchaudio
from torchaudio.transforms import MFCC
from sklearn.preprocessing import StandardScaler

def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    # Adjust n_mels here if necessary, and ensure n_fft and hop_length are appropriate for your sample rate
    mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=12, melkwargs={'n_mels': 40, 'n_fft': 400, 'hop_length': 160})
    mfcc = mfcc_transform(waveform)

    # Average across channels if stereo audio
    if mfcc.dim() > 2:
        mfcc = mfcc.mean(dim=0)

    # Standardize features
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc.numpy())

    # Swap dimensions: We want mfcc features in dim=-1
    result = np.transpose(mfcc_scaled)

    return result

def segment_audio(mfcc, segment_length):
    num_segments = mfcc.shape[0] // segment_length
    segments = mfcc[:num_segments*segment_length].reshape(num_segments, segment_length, -1)
    return segments

import os
import multiprocessing
from multiprocessing import Pool

def dataset_process_file(file_path):
    processed_audio = preprocess_audio(file_path)
    # +1 because we remove one element in __getitem__ below
    parts = segment_audio(processed_audio, args.segment_length + 1)
    return parts

class AudioSegmentDataset(Dataset):
    def __init__(self, args):

        audio_files = []
        for filename in os.listdir(args.dir):
            # torchaudio does not have an API to check if a file extension is supported
            if filename.lower().endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(args.dir, filename))

        print(f"Loading {len(audio_files)} audio files..")

        # Use multiprocessing Pool to process files in parallel
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            all_parts = pool.map(dataset_process_file, audio_files)

        # Combine parts into one dataset
        self.segments = np.concatenate(all_parts, axis=0)

        print(f"Loaded dataset shape: {self.segments.shape}")

    def get_feature_dim(self):
        return self.segments.shape[2]

    def __len__(self):
        return self.segments.shape[0]

    def __getitem__(self, idx):
        # Segment shape: (sequence_length, num_features)
        segment = self.segments[idx]

        # Input features: All time steps except the last
        input_features = torch.tensor(segment[:-1], dtype=torch.float32)
        
        # Target features: All time steps except the first
        target_features = torch.tensor(segment[1:], dtype=torch.float32)

        return input_features, target_features

def generate_audio_datasets(args):
    # Convert list of segments into a dataset
    dataset = AudioSegmentDataset(args)
    mfcc_feature_dim = dataset.get_feature_dim()

    # Split the dataset into training and validation
    total_size = len(dataset)
    validation_size = int(total_size * 0.25)

    # Generate random indices for validation, the rest are for training
    all_indices = np.arange(total_size)
    np.random.shuffle(all_indices)
    validation_indices = all_indices[:validation_size]
    train_indices = np.sort(all_indices[validation_size:])

    # We ensure each of the indices splits are from consecutive parts.
    # The idea is that different songs in the dataset are consecutive,
    # so this splits the dataset into separate playlists.
    part_size = len(train_indices) // 3
    train_indices_1 = train_indices[:part_size]
    train_indices_2 = train_indices[part_size:2*part_size]
    train_indices_3 = train_indices[2*part_size:]

    # Create DataLoaders
    # Note: num_workers=4 and/or pin_memory=True do not improve training throughput
    train_loader_full = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    train_loader_1 = DataLoader(Subset(dataset, train_indices_1), batch_size=args.batch_size, shuffle=True)
    train_loader_2 = DataLoader(Subset(dataset, train_indices_2), batch_size=args.batch_size, shuffle=True)
    train_loader_3 = DataLoader(Subset(dataset, train_indices_3), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, validation_indices), batch_size=args.batch_size, shuffle=False)

    return [train_loader_1, train_loader_2, train_loader_3, train_loader_full], val_loader, mfcc_feature_dim


################################################################################
# Training Loop

import torch.optim as optim
from tqdm.auto import tqdm

def calculate_average_loss(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    criterion = nn.MSELoss()

    sum_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            sum_loss += loss.item()

    avg_loss = sum_loss / len(data_loader)

    return avg_loss

from torch.optim.lr_scheduler import (SequentialLR, LinearLR, CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, StepLR, MultiStepLR,
                                      ExponentialLR, OneCycleLR)

def build_lr_scheduler(optimizer, scheduler_type, warmup_epochs, total_epochs, **kwargs):
    warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

    if scheduler_type == "StepLR":
        scheduler = StepLR(optimizer, step_size=kwargs.get('step_size', 50), gamma=kwargs.get('gamma', 0.5))
    elif scheduler_type == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=kwargs.get('milestones', [30, 60]), gamma=kwargs.get('gamma', 0.1))
    elif scheduler_type == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.9))
    elif scheduler_type == "OneCycleLR":
        scheduler = OneCycleLR(optimizer, max_lr=kwargs.get('max_lr', 0.01), total_steps=total_epochs+1)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=kwargs.get('T_0', total_epochs - warmup_epochs), T_mult=kwargs.get('T_mult', 1))
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    combined_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, scheduler], milestones=[warmup_epochs])

    return combined_scheduler

def train(model, train_loader, val_loader, args, fimmer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.mgpu:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    if args.compile:
        model = torch.compile(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args.scheduler, args.warmup_epochs, args.epochs)

    # Wrap the epoch range with tqdm
    epochs_tqdm = tqdm(range(args.epochs), desc='Overall Progress', leave=True)

    for epoch in epochs_tqdm:
        model.train()

        sum_train_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Reset gradients for each batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if fimmer:
                loss = loss + fimmer.loss()
            loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()  # Adjust model weights
                optimizer.zero_grad()  # Reset gradients tensors

            sum_train_loss += loss.item()

        avg_train_loss = sum_train_loss / len(train_loader)

        avg_val_loss = calculate_average_loss(model, val_loader)

        scheduler.step()

        # Update tqdm description for epochs with average loss
        epochs_tqdm.set_description(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        epochs_tqdm.refresh()  # to show immediately the update

    print(f"\nFinal Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}\n")


################################################################################
# Entrypoint

import argparse, random, csv, os

def seed_random(seed):
    if seed == 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_weights(model):
    for name, param in model.named_parameters():
        print(f"{name}:")
        print(param.data)

def log_experiment_results(args, train_loss_0, train_loss_1, train_loss_2, train_loss_full, val_loss):
    file_exists = os.path.isfile(args.log_file)
    with open(args.log_file, 'a', newline='') as csvfile:
        fieldnames = ['alpha', 'beta', 'train_loss_0', 'train_loss_1', 'train_loss_2', 'train_loss_full', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Only write the header once

        writer.writerow({
            'alpha': args.alpha,
            'beta': args.beta,
            'train_loss_0': train_loss_0,
            'train_loss_1': train_loss_1,
            'train_loss_2': train_loss_2,
            'train_loss_full': train_loss_full,
            'val_loss': val_loss
        })

def main(args):
    seed_random(args.seed)

    train_loaders, val_loader, input_dim = generate_audio_datasets(args)

    if args.full:
        # Full experiment:

        seed_random(args.seed)

        model_full = AudioS4(dim=input_dim)

        train(model_full, train_loaders[3], val_loader, args)

        # Split experiment:

        seed_random(args.seed)

        model_split = AudioS4(dim=input_dim)

        train(model_split, train_loaders[0], val_loader, args)
        train(model_split, train_loaders[1], val_loader, args)
        train(model_split, train_loaders[2], val_loader, args)

    # Split experiment with FIM:

    seed_random(args.seed)

    model_fim = AudioS4(dim=input_dim)

    train(model_fim, train_loaders[0], val_loader, args, fimmer=None)
    fimmer = FIMRegularizedTraining(model_fim, alpha=args.alpha)
    train(model_fim, train_loaders[1], val_loader, args, fimmer)
    fimmer.accumulate_model(beta=args.beta)
    train(model_fim, train_loaders[2], val_loader, args, fimmer)

    # Evaluation:

    print(f"Model parameters: {count_parameters(model_fim)}")

    if args.full:
        avg_train_loss_0 = calculate_average_loss(model_full, train_loaders[0])
        avg_train_loss_1 = calculate_average_loss(model_full, train_loaders[1])
        avg_train_loss_2 = calculate_average_loss(model_full, train_loaders[2])
        avg_train_loss_full = calculate_average_loss(model_full, train_loaders[3])
        avg_val_loss = calculate_average_loss(model_full, val_loader)

        print(f"Full Dataset: Final training loss: 0={avg_train_loss_0} 1={avg_train_loss_1} 2={avg_train_loss_2}")
        print(f"Full Dataset: Final training loss: {avg_train_loss_full}")
        print(f"Full Dataset: Final validation loss: {avg_val_loss}")

        avg_train_loss_0 = calculate_average_loss(model_split, train_loaders[0])
        avg_train_loss_1 = calculate_average_loss(model_split, train_loaders[1])
        avg_train_loss_2 = calculate_average_loss(model_split, train_loaders[2])
        avg_train_loss_full = calculate_average_loss(model_split, train_loaders[3])
        avg_val_loss = calculate_average_loss(model_split, val_loader)

        print(f"3x Split Dataset: Final training loss: 0={avg_train_loss_0} 1={avg_train_loss_1} 2={avg_train_loss_2}")
        print(f"3x Split Dataset: Final training loss: {avg_train_loss_full}")
        print(f"3x Split Dataset: Final validation loss: {avg_val_loss}")

    avg_train_loss_0 = calculate_average_loss(model_fim, train_loaders[0])
    avg_train_loss_1 = calculate_average_loss(model_fim, train_loaders[1])
    avg_train_loss_2 = calculate_average_loss(model_fim, train_loaders[2])
    avg_train_loss_full = calculate_average_loss(model_fim, train_loaders[3])
    avg_val_loss = calculate_average_loss(model_fim, val_loader)

    print(f"FIM Dataset: Final training loss: 0={avg_train_loss_0} 1={avg_train_loss_1} 2={avg_train_loss_2}")
    print(f"FIM Dataset: Final training loss: {avg_train_loss_full}")
    print(f"FIM Dataset: Final validation loss: {avg_val_loss}")

    if args.print_weights:
        print_model_weights(model_fim)

    if args.log_file:
        log_experiment_results(args, avg_train_loss_0, avg_train_loss_1, avg_train_loss_2, avg_train_loss_full, avg_val_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an RNN on audio data for next-sequence prediction')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Accumulation steps for gradients')
    parser.add_argument('--segment_length', type=int, default=1024, help='Input segment size')
    parser.add_argument('--seed', type=int, default=42, help='Seed for randomization of data loader')
    parser.add_argument('--dir', type=str, default="./data", help='Directory to scan for audio files')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--mgpu', action='store_true', help='Enable multi-GPU training')
    parser.add_argument('--print_weights', action='store_true', help='Print weights at the end to check them')
    parser.add_argument('--full', action='store_true', help='Full comparison')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for FIM loss term')
    parser.add_argument('--beta', type=float, default=0.5, help='FIM EWMA decay rate')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Warmup epochs')
    parser.add_argument('--scheduler', type=str, default="CosineAnnealingWarmRestarts", help='Scheduler')
    parser.add_argument('--log_file', type=str, default="", help='Output file for experiment results')
    args = parser.parse_args()

    main(args)
