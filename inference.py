import torch
import matplotlib.pyplot as plt
from .utils import clip_image, compute_psnr, compute_ssim
from .config import chosen_degradation as classes

def infer(degrade_img, daffir, guided_filter, device):
    # Parse data and move to device
    degrade_img = degrade_img.to(device)
    # Set daffirs to evaluation mode
    daffir.eval()
    # Forward through DAFFIR
    with torch.no_grad():
        output = daffir(degrade_img)
        d_c = output['degrade_outputs'][0]  # Degradation classification output
        d_l = output['degrade_outputs'][1]
        recon_img = output['restore_outputs']  # Restored image output
        input_patches = output['input_patches']
        # Apply guided filter
        gf_img = guided_filter(recon_img, input_patches, d_c, d_l)
        recon_img = clip_image(recon_img)
        gf_img = clip_image(gf_img)
    return recon_img, gf_img, d_c, d_l

def visualize_results(img, degrade_img, recon_img, gf_img, degrade_label, d_c, sample_num=4, return_fig=False):
    # Compute metrics
    recover_imgs = [recon_img, gf_img]
    psnr_results = [compute_psnr(img, recover_img) for recover_img in recover_imgs]
    ssim_results = [compute_ssim(img, recover_img) for recover_img in recover_imgs]

    # Visualization
    fig, ax = plt.subplots(
        sample_num, 2 + len(recover_imgs), figsize=(15, 5 * sample_num),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.75},
    )
    fig.suptitle('Real / Degrade / GF / Clean / Residual (PSNR / SSIM)', fontsize=24, y=1.01)
    fig.subplots_adjust(top=0.95)
    for i in range(sample_num):
        ## Original image
        ax[i, 0].imshow(img[i].permute(1, 2, 0).numpy())
        ax[i, 0].axis('off')
        
        ## Degrade image
        ax[i, 1].imshow(degrade_img[i].permute(1, 2, 0).numpy())
        pred_class = classes[torch.argmax(d_c[i][:-1]).numpy()]
        true_class = classes[torch.argmax(degrade_label[i]).numpy()]
        ax[i, 1].set_title(f'PRED: {pred_class} / {true_class}', fontsize=16)
        ax[i, 1].axis('off')
        
        ## Clean reconstruction
        ax[i, 2].imshow(recon_img[i].permute(1, 2, 0).numpy())
        ax[i, 2].set_title(f"{float(psnr_results[0][i]):.2f}/{float(ssim_results[0][i]):.2f}", fontsize=16)
        ax[i, 2].axis('off')

        ## Guided filter reconstruction
        ax[i, 3].imshow(gf_img[i].permute(1, 2, 0).numpy())
        ax[i, 3].set_title(f"{float(psnr_results[1][i]):.2f}/{float(ssim_results[1][i]):.2f}", fontsize=16)
        ax[i, 3].axis('off')
    if return_fig:
        return fig
    else:
        plt.show()