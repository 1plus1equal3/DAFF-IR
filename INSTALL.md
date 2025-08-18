# Installation

### Dependencies Installation

Follow these instructions

1. Clone our repository
```
git clone https://github.com/1plus1equal3/DAFF-IR.git
cd DAFF-IR
```

2. Install dependencies
```
pip install -r requirements.txt
```

### Dataset Download and Preparation

All the 5 datasets used in the paper can be downloaded from the following locations:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://drive.google.com/file/d/19_mCE_GXfmE5yYsm-HEzuZQqmwMjPpJr/view?usp=sharing), [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) (OTS)

Deblur: [Gopro](https://seungjunnah.github.io/Datasets/gopro)

Low-light Enhancement: [LOL](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view)

The training data should be placed in ```{root}/{task_name}/train``` directory where ```root``` is the directory that you want to store the data, and ```task_name``` can be ```haze```, ```new_if_dataset/blur```, ```new_if_dataset/low_light```, ```new_if_dataset/noise```, ```new_if_dataset/rain```. Similarly, the testing data should be placed in ```{data}/{task_name}/test```.
After placing the training data and testing the directory structure would be as follows:
```
├───haze
│    ├───train
│    │    ├───clear
│    │    └───haze
│    └───test
│         ├───gt
│         └───hazy
└───new_if_dataset
     ├───train
     │    ├───blur
     │    │     ├───0
     │    │     └───1
     │    ├───low_light
     │    │     ├───0
     │    │     └───1
     │    ├───noise
     │    │     ├───BSD400
     │    │     └───WaterlooED
     │    └───rain
     │          ├───0
     │          └───1
     └───test
          ├───blur
          │     ├───0
          │     └───1
          ├───low_light
          │     ├───0
          │     └───1
          ├───noise
          │     ├───CBSD68
          │     └───urban100
          └───rain
                ├───0
                └───1
```
