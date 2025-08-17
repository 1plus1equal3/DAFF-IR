import glob
import os

def get_haze_paths(haze_base_path, clear_base_path, train=True):

    haze_paths = []
    clear_paths = []

    for haze_path in glob.glob(haze_base_path + '/*/*.*' if train else haze_base_path + '/*.*'):
        haze_name = os.path.basename(haze_path)
        clear_name = haze_name.split('_')[0] + ('.jpg' if train else '.png')
        clear_path = os.path.join(clear_base_path, clear_name)
        assert os.path.exists(clear_path), f"File: {clear_name} does not exist"
        haze_paths.append(haze_path)
        clear_paths.append(clear_path)
    
    assert len(haze_paths) == len(clear_paths), "Mismatch in number of haze and clear images"
    return haze_paths, clear_paths

def get_other_paths(path):

    degrade_type = os.listdir(path)
    print(degrade_type)
    path_ds = {}
    for d in degrade_type:
        if d == 'noise':
            img_paths = glob.glob(os.path.join(path, d) + '/**/*.*', recursive=True)
            degrade_img_paths = glob.glob(os.path.join(path, d) + '/**/*.*', recursive=True)
        else:
            img_paths = glob.glob(os.path.join(path, d, '1') + '/*.*')
            degrade_img_paths = glob.glob(os.path.join(path, d, '0') + '/*.*')
        # Sort paths
        img_paths.sort()
        degrade_img_paths.sort()
        path_ds[d] = (img_paths, degrade_img_paths)
    return path_ds

def get_all_paths(other_path, haze_base_path, clear_base_path, train=True):
    # Get the paths for training images and their degraded versions
    haze_paths, clear_paths = get_haze_paths(haze_base_path, clear_base_path, train=train)
    path_ds = get_other_paths(other_path)
    # Add haze training images path
    path_ds['haze'] = (clear_paths, haze_paths)
    print('Training list' if train else 'Testing list')
    for k in path_ds.keys():
        print(f'Degradation {k}: {len(path_ds[k][0])} images')
    return path_ds