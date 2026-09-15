#!/usr/bin/env python
# coding: utf-8



# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import os
import sys
import tempfile
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
sys.path.append('/home/sherpa/NbFold')
import nbfold
from nbfold.models.AbResNet import AbResNet
from nbfold.util.masking import MASK_VALUE
from nbfold.util.training import check_for_h5_file
from nbfold.datasets.H5PairwiseGeometryDataset import H5PairwiseGeometryDataset
from nbfold.preprocess.generate_h5_pairwise_geom_file import antibody_to_h5

# ---------------------------------------------------------------------
# CUDA performance flags
# ---------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
_output_names = ['ca_dist', 'cb_dist', 'no_dist', 'omega', 'theta', 'phi']

num_blocks1D = 4
num_blocks2D = 16
dilation_cycle = 5
num_bins = 37
dropout = 0.4

epochs = 400
batch_size = 2 #20 #16
lr = 0.01
train_split = 0.9
gradient_clip_val = 0.5
random_seed = 0

NUM_WORKERS = 100
PREFETCH_FACTOR = 4
save_every = 10

# ---------------------------------------------------------------------
# Model metadata (CRITICAL for inference)
# ---------------------------------------------------------------------
MODEL_META = {
    'input_dim': 21,
    'num_out_bins': num_bins,
    'num_blocks1D': num_blocks1D,
    'num_blocks2D': num_blocks2D,
    'dropout_proportion': dropout,
    'dilation_cycle': dilation_cycle
}

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
project_path = os.path.abspath(os.path.join(nbfold.__file__, '../..'))
h5_file = os.path.join(project_path, 'data/final_antibody.h5')
antibody_database = os.path.join(project_path, 'data/antibody_database')

now = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
output_dir = os.path.join(project_path, 'trained_models', f'model_{now}')
os.makedirs(output_dir, exist_ok=True)

ckpt_dir = os.path.join(project_path, 'trained_models', 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, 'latest_checkpoint.pt')

# ---------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------
def atomic_save(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.pt')
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)

# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
# check_for_h5_file(h5_file, antibody_to_h5, antibody_database)

dataset = H5PairwiseGeometryDataset(
    h5_file,
    num_bins=num_bins,
    mask_distant_orientations=True
)

train_len = int(len(dataset) * train_split)
torch.manual_seed(random_seed)
train_ds, val_ds = data.random_split(
    dataset, [train_len, len(dataset) - train_len]
)

train_loader = data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=PREFETCH_FACTOR,
    collate_fn=H5PairwiseGeometryDataset.merge_samples_to_minibatch
)

val_loader = data.DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=PREFETCH_FACTOR,
    collate_fn=H5PairwiseGeometryDataset.merge_samples_to_minibatch
)

print(f'Dataset: {len(dataset)} | Train: {len(train_ds)} | Val: {len(val_ds)}')

# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, ignore_index=MASK_VALUE):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

criterion = FocalLoss(ignore_index=dataset.mask_fill_value)

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
model = AbResNet(
    MODEL_META['input_dim'],
    num_out_bins=MODEL_META['num_out_bins'],
    num_blocks1D=MODEL_META['num_blocks1D'],
    num_blocks2D=MODEL_META['num_blocks2D'],
    dropout_proportion=MODEL_META['dropout_proportion'],
    dilation_cycle=MODEL_META['dilation_cycle']
).to(device)

if hasattr(torch, "compile"):
    model = torch.compile(model, mode="max-autotune")
    print(" torch.compile enabled")

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))

# ---------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------
best_val = float('inf')
start_epoch = 0

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    best_val = ckpt['best_val_loss']
    start_epoch = ckpt['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

# ---------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------
def run_epoch(loader, training=True):
    model.train(training)
    total_loss = torch.zeros(len(_output_names) + 1)

    for inputs, e, max_len, labels in tqdm(loader, leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = [l.to(device, non_blocking=True) for l in labels]

        emb = torch.stack(e).to(device, non_blocking=True)
        max_len = torch.as_tensor(max_len, dtype=torch.long, device=device)
        emb = emb[:, :, :max_len.max()]

        with torch.set_grad_enabled(training):
            outputs = model(inputs, emb)
            losses = [criterion(o, l) for o, l in zip(outputs, labels)]
            loss = sum(losses)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                optimizer.step()

        losses.append(loss.detach())
        total_loss += torch.tensor([l.item() for l in losses])

    return total_loss / len(loader)

# ---------------------------------------------------------------------
# Save inference checkpoint
# ---------------------------------------------------------------------
def save_inference_checkpoint(path):
    torch.save({
        **MODEL_META,
        'model_state_dict': model.state_dict()
    }, path)

# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
for epoch in range(start_epoch, epochs):
    print(f"\nEpoch {epoch}/{epochs-1}")

    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)

    train_dict = dict(zip(_output_names + ['total'], train_loss.tolist()))
    val_dict = dict(zip(_output_names + ['total'], val_loss.tolist()))

    writer.add_scalars('train_loss', train_dict, epoch)
    writer.add_scalars('val_loss', val_dict, epoch)

    print('Train:', train_dict)
    print('Val  :', val_dict)

    scheduler.step(val_dict['total'])

    # Save best inference model
    if val_dict['total'] < best_val:
        best_val = val_dict['total']
        save_inference_checkpoint(
            os.path.join(output_dir, 'best_model.pt')
        )

    # Save resume checkpoint
    atomic_save({
        'epoch': epoch,
        'best_val_loss': best_val,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, ckpt_path)

    # Optional snapshots
    if (epoch + 1) % save_every == 0:
        save_inference_checkpoint(
            os.path.join(output_dir, f'model_epoch_{epoch}.pt')
        )

writer.close()
print("Training complete.")
