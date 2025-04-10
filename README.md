## IGMamba-Net: Inverse External with Gated Attention Driven Mamba-CNN for Biomedical Image Segmentation
## Introduction
IGMamba is a general multi-network model based on the Mamba-CNN for biomedical image segmentation.We introduced inverse external attention and gated attention, and based on them designed a local feature extraction module and a feature fusion extraction module, and introduced a global feature extraction module to enhance the segmentation performance.This repository contains the code used in the study.
## Installation
```bash
git clone https://github.com/Brainsmatics/IGMamba
```
```
pip install -r requirements.txt
```
## Prepare the dataset
```text
./data/filename/
├── train
│   ├── images
│   │   └── *.png
│   └── masks
│       └── *.png
└── test
    ├── images
    │   └── *.png
    └── masks
        └── *.png
```
## Train and Test
```
cd IGMamba  
python train.py  
python test.py 
```
## Acknowledgement
Some code is reused from the([VM-UNet](https://github.com/JCruan519/VM-UNet))、([External_Attention](https://arxiv.org/abs/2105.02358))、([MALUNet](https://github.com/JCruan519/MALUNet)).
