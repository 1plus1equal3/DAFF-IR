import os 
import sys
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from transforms import crop_img, random_crop, random_augmentation, gauss_noise, degrade_regularization, get_label
from torch.utils.data import ConcatDataset

class DegradationDataset(Dataset):
    def __init__(self, norm_paths, degrade_paths, degrade_reg=False, noise_sigma=25, crop_size=128, chosen_degradation=None):
        self.norm_paths = norm_paths
        self.degrade_paths = degrade_paths
        self.degrade_reg = degrade_reg
        self.noise_sigma = noise_sigma
        self.crop_size = crop_size
        self.chosen_degradation = chosen_degradation

    def __len__(self):
        return len(self.norm_paths)

    def __getitem__(self, idx):
        # Get paths for normal and degraded images
        norm_path = self.norm_paths[idx]
        degrade_path = self.degrade_paths[idx]
        # Convert paths to tensors
        img = TF.to_tensor(crop_img(Image.open(norm_path).convert('RGB'), base=32))
        degrade_img = TF.to_tensor(crop_img(Image.open(degrade_path).convert('RGB'), base=32))
        # Apply augmentation
        img, degrade_img = random_crop(img, degrade_img, self.crop_size)
        if degrade_path.split(os.sep)[-3] == 'noise':
            degrade_img = gauss_noise(degrade_img, self.noise_sigma)
        img, degrade_img = random_augmentation(img, degrade_img, random.randint(0, 7))
        # Get labels
        label = get_label(degrade_path, -4 if degrade_path.split(os.sep)[-3] == 'test' else -3, self.noise_sigma, self.chosen_degradation)
        # If use degrade regularization
        if self.degrade_reg:
            degrade_level = random.uniform(0.25, 1.0)
            noise_level = 0.0
            degrade_img, residual, reg_residual = degrade_regularization(degrade_img, img, degrade_level, noise_level)
        else:
            degrade_level = noise_level = 0.0

        return degrade_img, img, label, torch.tensor(degrade_level), torch.tensor(noise_level)

def build_dataset(train_path_ds, test_path_ds, chosen_degradation, degrade_type=None):
    # Build Dataset Lists
    train_datasets = []
    test_datasets = []
    max_len_train = max(len(train_path_ds['noise'][0]) if 'noise' in k else len(train_path_ds[k][0]) for k in chosen_degradation)
    # max_len_test = max(len(test_path_ds['noise'][0]) if 'noise' in k else len(test_path_ds[k][0]) for k in classes)

    # Create train datasets
    for k in chosen_degradation:
        # Ensure the train and test datasets have the same length for each degradation type
        if 'noise' in k:
            train_reps = max_len_train//len(train_path_ds['noise'][0])
            # print(f'Repeating number for {k}: {train_reps}')
            noise_sigma = int(k.split('_')[1])
            train_paths = [train_path_ds['noise'][0]*train_reps, train_path_ds['noise'][1]*train_reps] 
            test_paths = test_path_ds['noise']
        else:
            train_reps = max_len_train//len(train_path_ds[k][0])
            # print(f'Repeating number for {k}: {train_reps}')
            noise_sigma = 0
            train_paths = [train_path_ds[k][0]*train_reps, train_path_ds[k][1]*train_reps] 
            test_paths = test_path_ds[k]
        train_datasets.append(DegradationDataset(*train_paths, degrade_reg=False, noise_sigma=noise_sigma, chosen_degradation=chosen_degradation))
        test_datasets.append(DegradationDataset(*test_paths, degrade_reg=False, noise_sigma=noise_sigma, chosen_degradation=chosen_degradation))

    if degrade_type is None:
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)
    else:
        index = chosen_degradation.index(degrade_type)
        train_dataset = train_datasets[index]
        test_dataset = test_datasets[index]

    # Check dataset length
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Test dataset length: {len(test_dataset)}')

    return train_dataset, test_dataset