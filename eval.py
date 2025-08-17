import torch
from tqdm import tqdm
from utils.metrics import Metrics
from utils.loss import *
from config import *
from utils.utils import *
from utils.metrics import Metrics
from model import DAFFIR, DAGF

def evaluate(self, ds_test, daffir: DAFFIR, guided_filter: DAGF, metric: Metrics, device: str, eval_epoch=0):    
    if device:
        self.device = device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set models to evaluation mode
    daffir.eval()
    guided_filter.eval()
    
    for deg_type in chosen_degradation:
        metric.psnr_metrics[deg_type].reset()
        metric.ssim_metrics[deg_type].reset()
        metric.gf_psnr_metrics[deg_type].reset()
        metric.gf_ssim_metrics[deg_type].reset()
    
    degradation_counts = {deg_type: 0 for deg_type in chosen_degradation}
    with torch.no_grad():
        for batch in tqdm(ds_test):
            # Unpack batch and move to device
            degrade_img = batch[0].to(device)
            img = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            # Forward through DAFFIR
            output = self.daffir(degrade_img)
            d_c = output['degrade_outputs'][0]  # Degradation classification output
            d_l = output['degrade_outputs'][1]  # Degradation level output
            recon_img = output['restore_outputs']  # Restored image output
            input_patches = output['input_patches']  # Input patches for guided filter
            recon_img = clip_image(recon_img)  # Restored image output
            ## Apply guided filter
            gf_img = clip_image(guided_filter(recon_img, input_patches, d_c, d_l))
            
            batch_size = img.shape[0]
            for i in range(batch_size):
                deg_idx = torch.argmax(labels[i], dim=-1).item()
                deg_type = chosen_degradation[deg_idx]
                # Update metrics for this degradation type with DAFFIR output
                metric.psnr_metrics[deg_type].update(recon_img[i:i+1], img[i:i+1])
                metric.ssim_metrics[deg_type].update(recon_img[i:i+1], img[i:i+1])
                # Update metrics for this degradation type with Guided Filter output
                metric.gf_psnr_metrics[deg_type].update(gf_img[i:i+1], img[i:i+1])
                metric.gf_ssim_metrics[deg_type].update(gf_img[i:i+1], img[i:i+1])
                degradation_counts[deg_type] += 1
    
    print("\n--- Evaluation Results ---")
    overall_psnr_sum = 0
    overall_ssim_sum = 0
    gf_overall_psnr_sum = 0
    gf_overall_ssim_sum = 0
    overall_count = 0
    
    for deg_type in chosen_degradation:
        if degradation_counts[deg_type] > 0:
            # Compute average PSNR and SSIM with DAFFIR output
            avg_psnr = metric.psnr_metrics[deg_type].compute().item()
            avg_ssim = metric.ssim_metrics[deg_type].compute().item()
            # Compute average PSNR and SSIM with Guided Filter output
            gf_avg_psnr = metric.gf_psnr_metrics[deg_type].compute().item()
            gf_avg_ssim = metric.gf_ssim_metrics[deg_type].compute().item()
            print(f"{deg_type.upper()} - PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.2f}, Count: {degradation_counts[deg_type]}")
            print(f"{deg_type.upper()} (GF) - PSNR: {gf_avg_psnr:.2f}, SSIM: {gf_avg_ssim:.2f}")
            # Log metrics to wandb
            metric_dict = {
                f'eval_psnr_{deg_type}': avg_psnr,
                f'eval_ssim_{deg_type}': avg_ssim,
                f'eval_psnr_gf_{deg_type}': gf_avg_psnr,
                f'eval_ssim_gf_{deg_type}': gf_avg_ssim,
            }
            self.wandb_log_metric(metric_dict)
            # Prepare overall metrics
            overall_psnr_sum += avg_psnr * degradation_counts[deg_type]
            overall_ssim_sum += avg_ssim * degradation_counts[deg_type]
            gf_overall_psnr_sum += gf_avg_psnr * degradation_counts[deg_type]
            gf_overall_ssim_sum += gf_avg_ssim * degradation_counts[deg_type]
            overall_count += degradation_counts[deg_type]
    # Compute mean overall PSNR and SSIM
    avg_overall_psnr = overall_psnr_sum / overall_count if overall_count else 0
    avg_overall_ssim = overall_ssim_sum / overall_count if overall_count else 0
    avg_gf_overall_psnr = gf_overall_psnr_sum / overall_count if overall_count else 0
    avg_gf_overall_ssim = gf_overall_ssim_sum / overall_count if overall_count else 0
    print("\n--- Overall Results ---")
    print(f"OVERALL - PSNR: {avg_overall_psnr:.2f}, SSIM: {avg_overall_ssim:.2f}")
    print(f"OVERALL (GF) - PSNR: {avg_gf_overall_psnr:.2f}, SSIM: {avg_gf_overall_ssim:.2f}")
    # Log overall metrics to wandb
    self.wandb_log_metric({
        'eval_psnr_overall': avg_overall_psnr,
        'eval_ssim_overall': avg_overall_ssim,
        'eval_psnr_gf_overall': avg_gf_overall_psnr,
        'eval_ssim_gf_overall': avg_gf_overall_ssim,
    })
    # Reset models to training mode
    self.daffir.train()
    self.guided_filter.train()