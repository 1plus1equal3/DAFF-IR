import os
import torch
import torchmetrics
from PIL import Image
import numpy as np
import sys
from config import CKPT_DIR, PROJECT_NAME

# Create checkpoint objects
checkpoint_dir = CKPT_DIR
project = PROJECT_NAME
checkpoint_prefix = f"{checkpoint_dir}/{project}"

# Save checkpoint function
def save_checkpoint(epoch, daff_ir, guided_filter, optimizer, scheduler=None, gf_optimizer=None, gf_scheduler=None):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = f"{checkpoint_prefix}_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'daff_ir_state_dict': daff_ir.state_dict(),
        'guided_filter_state_dict': guided_filter.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'gf_optimizer_state_dict': gf_optimizer.state_dict(),
        'gf_scheduler_state_dict': gf_scheduler.state_dict() if gf_scheduler else None,
    }, checkpoint_path)
    print(f"Saved checkpoint for Epoch {epoch}: {checkpoint_path}")


def restore_checkpoint(checkpoint_path, daff_ir, guided_filter, optimizer, scheduler=None, gf_optimizer=None, gf_scheduler=None):
    if checkpoint_path is None:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(project) and f.endswith('.pt')]
        if not checkpoints:
            print("No checkpoint found to restore.")
            return 0
        epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints if '_epoch_' in f]
        if not epochs:
            print("No valid checkpoint format found.")
            return 0
        latest_epoch = max(epochs)
        checkpoint_path = f"{checkpoint_prefix}_epoch_{latest_epoch}.pt"

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            daff_ir.load_state_dict(checkpoint['daff_ir_state_dict'])
            guided_filter.load_state_dict(checkpoint['guided_filter_state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            gf_optimizer.load_state_dict(checkpoint['gf_optimizer_state_dict'])
            if 'gf_scheduler_state_dict' in checkpoint and gf_scheduler is not None:
                gf_scheduler.load_state_dict(checkpoint['gf_scheduler_state_dict'])

            start_epoch = checkpoint.get('epoch', 0)
            print(f"Checkpoint restored from {checkpoint_path}, starting from epoch {start_epoch}")
            return start_epoch
        except Exception as e:
            print(f"Error restoring checkpoint: {e}")
            return 0
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0
    
## Utils functions
def clip_image(image):
    return torch.clamp(image, 0., 1.)

def compute_psnr(orig, pred):
    """Compute PSNR per image in batch (no reduction)"""
    orig = clip_image(orig)  # assumes [B, C, H, W] and values in [0,1]
    pred = clip_image(pred)
    psnr = []
    for i in range(orig.shape[0]):
        psnr.append(
            torchmetrics.functional.peak_signal_noise_ratio(
                pred[i].unsqueeze(0), orig[i].unsqueeze(0), data_range=1.0
            )
        )
    return psnr


def compute_ssim(orig, pred):
    """Compute SSIM per image in batch (no reduction)"""
    orig = clip_image(orig)
    pred = clip_image(pred)
    ssim = []
    for i in range(orig.shape[0]):
        ssim.append(
            torchmetrics.functional.structural_similarity_index_measure(
                pred[i].unsqueeze(0), orig[i].unsqueeze(0), data_range=1.0
            )
        )
    return ssim

def crop_image(image, base=64):
    image = np.array(image)
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]

def load_image(image_path):
    """Load an image from a file and preprocess it."""
    image = Image.open(image_path).convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image

def save_image(image, save_path):
    """Save a tensor image to a file."""
    image = (image * 255).astype(np.uint8)  # Convert to [0, 255]
    Image.fromarray(image).save(save_path)

def logging(step, total_steps, metric_values):
    metric_str = f"\rStep {step+1}/{total_steps} - " + " - ".join(
        f"{name}: {value:.4f}" for name, value in metric_values.items()
    )
    sys.stdout.write(metric_str)
    sys.stdout.flush()