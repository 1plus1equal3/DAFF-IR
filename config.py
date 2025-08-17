import numpy as np
import os

# Training configuration
BASE_DATA_PATH = '/home/user02/linhdang/DAFF-IR/data/datasets/oneplusoneisthree/ir-5-degrade/versions/1'

# Train data path for haze
train_haze_base_path = f'{BASE_DATA_PATH}/haze/train/haze'
train_clear_base_path = f'{BASE_DATA_PATH}/haze/train/clear'

# Test data path for haze
test_haze_base_path = f'{BASE_DATA_PATH}/haze/test/hazy'
test_clear_base_path = f'{BASE_DATA_PATH}/haze/test/gt'

# Get list path for training and testing image
train_path = f'{BASE_DATA_PATH}/new_if_dataset/train'
test_path = f'{BASE_DATA_PATH}/new_if_dataset/test'

chosen_degradation = ['noise_15', 'noise_25', 'noise_50', 'haze', 'rain']

BATCH_SIZE = 32
WORKERS = 2
EPOCHS = 300
STEPS_PER_EPOCH = 1200
INIT_LR = 1e-4
WARMUP_LR = 2e-4
DECAY_LR = 1e-5
WARMUP_EPOCHS = 25
ACCUMULATE_GRADIENT_STEPS = 2

# Checkpoint configuration
CKPT_DIR = 'checkpoints'
PROJECT_NAME = 'DAFF-IR'
WANDB_KEY_PATHS = './wandb.txt' # Your wandb key

# Visualization and checkpoint configuration
VIS_EPOCH = 5
VIS_NUM = 8
SAVE_EPOCH = 5
EVAL_EPOCH = 20

# Latest checkpoint path
CKPT_PATH = '/home/user02/linhdang/DAFF-IR/v1_ckpt_2025-06-11/DAFF_LARGE_ckpt_epoch_339.pt'

num = 4

# Config for encoder 
encoder_config = {
    "num": 4,
    "modules_num": [2, 3, 3, 4],
    "window_size": list(16 // (2 ** np.arange(num))),
    "filters": list(48 * (2 ** np.arange(num))),
    "c_heads": list(2 ** np.arange(num)),
    # "s_heads": list(8 // (2 ** np.arange(num))),
    "s_heads": list(2 ** np.arange(num)),
    "expansion_factor": 2.66,  # For FuseFFN
    "bias": False,
    "degrade_class": False,
    "degrade_level": False,
    "shift_size": list(8 // (2 ** np.arange(num))),
    "input_shape": (3, None, None),
    "class_num": len(chosen_degradation), # The last index for clean image
}

# Config for decoder
decoder_config = {
    "num": num,
    "modules_num": encoder_config["modules_num"][::-1],
    "window_size": encoder_config["window_size"][::-1],
    "filters": encoder_config["filters"][::-1],
    "c_heads": encoder_config["c_heads"][::-1],
    "s_heads": encoder_config["s_heads"][::-1],
    "expansion_factor": encoder_config["expansion_factor"],
    "bias": encoder_config["bias"],
    "degrade_class": True,
    "degrade_level": False,
    "shift_size": encoder_config["shift_size"][::-1],
    "class_num": encoder_config['class_num'],
    "use_attr_fusion": True,  # Whether to use attribute fusion
}

# Config for refinement block
refine_config = {
    "modules_num": 4,
    "window_size": encoder_config["window_size"][0],
    "filters": encoder_config["filters"][0],
    "c_heads": encoder_config["c_heads"][0],
    "s_heads": encoder_config["s_heads"][0],
    "expansion_factor": encoder_config["expansion_factor"],
    "bias": encoder_config["bias"],
    "degrade_class": True,
    "degrade_level": False,
    "shift_size": encoder_config["shift_size"][0],
    "class_num": encoder_config['class_num'],
    "defocus": True,  # Whether to apply defocus filter
    "last_act": 'none',
}

# Guided Filter config
guided_filter_config = {
    "modules_num": 4,
    "window_size": 16,
    "filters": 48,
    "c_heads": 1,
    "s_heads": 1,
    "expansion_factor": 2.66,
    "bias": False,
    "degrade_class": True,
    "degrade_level": False,
    "shift_size": 8,  # Shift size for Swin Transformer
    "class_num": encoder_config['class_num'],
    "radius": 8,  # Radius for guided filter
}

# Training configuaration
restore_epoch = int(os.path.basename(CKPT_PATH).split('.')[0].split('_')[-1])
train_config = {
    "initial_epoch": restore_epoch if restore_epoch > 0 else 1,
    "gradient_accumulation_steps": 1,
    "vis_epoch": 5,
    "vis_num": 8,
    "save_epoch": 5
}

# Evaluate configuration
eval_config = {
    "eval_epoch": 5,
}