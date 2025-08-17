import os
import numpy as np 
import torch 
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop

def gauss_noise(image, sigma):
    noise = torch.normal(mean=0.0, std=1.0, size=image.shape)
    return torch.clamp(image + noise*sigma/255., 0., 1.)
    
def random_augmentation(img, degrade_img, mode):
    # Applies the same transformation to both images
    if mode == 0:
        # original
        out_img = img
        out_degrade = degrade_img
    elif mode == 1:
        # flip up and down
        out_img = TF.vflip(img)
        out_degrade = TF.vflip(degrade_img)
    elif mode == 2:
        # rotate 90 degrees counter-clockwise
        out_img = TF.rotate(img, 90)
        out_degrade = TF.rotate(degrade_img, 90)
    elif mode == 3:
        # rotate 90 and flip up and down
        out_img = TF.vflip(TF.rotate(img, 90))
        out_degrade = TF.vflip(TF.rotate(degrade_img, 90))
    elif mode == 4:
        # rotate 180 degrees
        out_img = TF.rotate(img, 180)
        out_degrade = TF.rotate(degrade_img, 180)
    elif mode == 5:
        # rotate 180 and flip up and down
        out_img = TF.vflip(TF.rotate(img, 180))
        out_degrade = TF.vflip(TF.rotate(degrade_img, 180))
    elif mode == 6:
        # rotate 270 degrees
        out_img = TF.rotate(img, 270)
        out_degrade = TF.rotate(degrade_img, 270)
    elif mode == 7:
        # rotate 270 and flip up and down
        out_img = TF.vflip(TF.rotate(img, 270))
        out_degrade = TF.vflip(TF.rotate(degrade_img, 270))
    else:
        raise ValueError('Invalid mode: must be an integer from 0 to 7')
    return out_img, out_degrade

def random_crop(img, degrade_img, size=128):
    i, j, h, w = RandomCrop.get_params(img, output_size=(size, size))
    return TF.crop(img, i, j, h, w), TF.crop(degrade_img, i, j, h, w)

def degrade_regularization(degrade_img, gt_img, degrade_level=0.5, noise_level=0.0):
    residual = degrade_img - gt_img
    degrade_noise = torch.empty_like(residual).uniform_(1 - noise_level, 1 + noise_level)
    reg_residual = residual * degrade_noise * degrade_level
    degrade_img = torch.clamp(gt_img + reg_residual, 0., 1.)
    return degrade_img, residual, reg_residual

def get_label(path, dir_index=-3, noise_sigma=0, chosen_degradation=None):
    label = path.split(os.sep)[dir_index]
    if noise_sigma>0:
        label += f'_{str(noise_sigma)}'
    index = chosen_degradation.index(label)
    return torch.nn.functional.one_hot(torch.tensor(index), num_classes=len(chosen_degradation)).to(torch.uint8)

def norm_data(image):
    image = image.float() / 255.0
    return image

def crop_img(image, base=64):
    image = np.array(image)
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]