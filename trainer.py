import torch
from model import DAFFIR, DAGF
from schedulers import WarmupCosineAnnealingLR
from metrics import Metrics
from .loss import *
from .config import *
from .utils import *
from wandb_logger import WandbLogger

def setup_optim(model, config):
    # Build optimizer and scheduler for the model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['init_lr'],
    )
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] * config['steps_per_epoch'],  # Total number of iterations
        warmup_epochs=config['warmup_epochs'] * config['steps_per_epoch'],  # Warmup iterations
        warmup_lr=config['warmup_lr'],  # Warmup learning rate
        eta_min=config['decay_lr'],  # Minimum learning rate
    )
    return optimizer, scheduler

class DaffIRTrainer:
    def __init__(self):
        # Set up device for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model initialize
        self.daffir = DAFFIR(encoder_config, decoder_config, refine_config).to(self.device)
        self.guided_filter = DAGF(guided_filter_config, refine_config).to(self.device)
        
        # Optimizer and scheduler initialize
        self.optimizer, self.scheduler = setup_optim(self.daffir)
        self.gf_optimizer, self.gf_scheduler = setup_optim(self.guided_filter)
        
        # Wandb logger initialize
        self.wandb_log_metric = WandbLogger()

        # Initialize metrics
        self.degrade_metrics = Metrics(name='degrade', mode='degrade', degrade_classes=chosen_degradation, device=self.device)
        self.restore_metrics = Metrics(name='base', mode='restore', degrade_classes=chosen_degradation, device=self.device)
        self.dagf_metrics = Metrics(name='dagf', mode='restore', degrade_classes=chosen_degradation, device=self.device)
        self.eval_metrics = Metrics(name='eval', mode='eval', degrade_classes=chosen_degradation, device=self.device)

    ## Train step for daffir generator
    def train_daffir(self, data, step, gradient_accumulation_steps=2):
        # Unpack data
        degrade_img, img, d_c, d_l, n_l = data
        degrade_img = degrade_img.to(self.device)
        img = img.to(self.device)
        d_c = d_c.to(self.device)
        d_l = d_l.to(self.device)

        self.daffir.train()  # Set DAFFIR to training mode
        self.guided_filter.train()  # Set guided filter to training mode
        
        #! Train DAFFIR generator
        # Forward pass through DAFFIR
        output = self.daffir(degrade_img)
        out_d_c = output['degrade_outputs'][0]  # Degradation classification output
        out_d_l = output['degrade_outputs'][1]  # Degradation level output
        recon_img = output['restore_outputs']  # Restored image output
        input_patches = output['input_patches']  # Input patches for guided filter

        # Compute losses for DAFFIR
        ## Degrade classification loss
        degrade_classify_loss = self.losses.classify_loss_fn(d_c, out_d_c) 
        degrade_level_loss = 0.0
        ## Reconstruction loss
        recon_loss = self.losses.charbonnier_loss(img, recon_img)
        ## Adversarial loss (if used)
        # Compute total loss
        total_loss = degrade_classify_loss + degrade_level_loss + recon_loss 
        total_loss = total_loss / gradient_accumulation_steps  # Normalize by gradient accumulation steps
        # Update gradients for DAFFIR
        total_loss.backward()
        daffir_params = list(self.daffir.parameters())
        torch.nn.utils.clip_grad_norm_(daffir_params, max_norm=1.0)
        if step % gradient_accumulation_steps == 0:
            # Step optimizer only after gradient accumulation
            self.optimizer.step()
            # Zero gradients
            self.optimizer.zero_grad()
        self.scheduler.step()  # Step the learning rate scheduler

        #! Train Guided filter
        # Forward pass through guided filter
        gf_img = self.guided_filter(torch.clamp(recon_img.detach(), 0., 1.), input_patches.detach(), out_d_c.detach(), out_d_l.detach())  # Guided filter output
        # Compute losses for guided filter
        gf_loss = self.losses.charbonnier_loss(img, gf_img)
        gf_loss = gf_loss / gradient_accumulation_steps  # Normalize by gradient accumulation steps
        # Update guided filter gradients
        gf_loss.backward()
        gf_params = list(self.guided_filter.parameters())
        torch.nn.utils.clip_grad_norm_(gf_params, max_norm=1.0)
        if step % gradient_accumulation_steps == 0:
            # Step optimizer only after gradient accumulation
            self.gf_optimizer.step()
            # Zero gradients
            self.gf_optimizer.zero_grad()
        self.gf_scheduler.step()  # Step the guided filter learning rate scheduler

        # Update metrics using torchmetrics
        with torch.no_grad():
            # Update metrics for degradation classification
            self.degrade_metrics.metrics['degrade_acc'].update(out_d_c, torch.argmax(d_c, dim=-1))
            self.degrade_metrics.metrics['degrade_loss'].update(degrade_classify_loss.item())
            # Update metrics for restoration
            self.restore_metrics.metrics[f'loss_{self.daffir.name}'].update(recon_loss.item())
            self.restore_metrics.metrics[f'psnr_{self.daffir.name}'].update(recon_img, img)
            self.restore_metrics.metrics[f'ssim_{self.daffir.name}'].update(recon_img, img)
            # Update metrics for guided filter
            self.dagf_metrics.metrics[f'loss_{self.guided_filter.name}'].update(gf_loss.item())
            self.dagf_metrics.metrics[f'psnr_{self.guided_filter.name}'].update(gf_img, img)
            self.dagf_metrics.metrics[f'ssim_{self.guided_filter.name}'].update(gf_img, img)

    def reset_metrics(self):
        # Reset all metrics
        self.degrade_metrics.reset()
        self.restore_metrics.reset()
        self.dagf_metrics.reset()
