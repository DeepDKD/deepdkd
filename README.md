# DeepDKD - Official Implementation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
Welcome to the official GitHub repository for DeepDKD, the implementation of our research paper: ​[Non-invasive biopsy diagnosis of diabetic kidney disease via deep learning applied to retinal images: a population-based study](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(25)00040-8/fulltext) (*The Lancet Digital Health, 2025*).

This project provides the codebase for:
1. Weakly supervised momentum contrastive learning for unsupervised representation learning from unlabeled retinal images
2. Training models to detect Diabetic Kidney Disease (DKD) from retinal fundus images
3. Differentiating isolated diabetic nephropathy (DN) from non-diabetic kidney disease (NDKD)
4. Comprehensive performance comparison with metadata model and combined model

## Project Configuration
Set up the environment using:
```bash
pip install -r requirements.txt
```

## Pretraining
The DeepDKD system used weakly supervised momentum contrastive learning as a method for extracting transferable visual representations of retinal fundus images with further supervised training in two tasks (detecting DKD and differentiating isolated from NDKD). 
Start pretraining with:
```bash
python pretrain.py
```
Make sure to customize the training parameters according to your dataset and requirements. 

## Training of the DeepDKD System
To train the DeepDKD system for the specific task, follow these steps:

1. Prepare your dataset: Organize your retinal fundus images dataset into appropriate training and test sets.
2. Configure training parameters: Adjust the hyperparameters in the training script according to your dataset characteristics and training requirements.
3. Execute the training script:

```bash
python train_dkd_cls.py
python train_dn_cls.py
```
Monitor the training process and evaluate the model on the validation set to ensure optimal performance.

## Metadata model & Combined model
For a comprehensive performance comparison of the DeepDKD system, we have developed two additional models: the Metadata Model and the Combined Model, based on XGBoost. 
To train these models and assess their performance, use the following steps:

1. Configure training parameters: Adjust the hyperparameters in the XGBoost training script (`train_xgb.py`) according to your dataset characteristics and evaluation requirements.
2. Execute the training script:

```bash
python train_xgb.py
```

## Citation

If you use this work, please cite:
```bibtex
@article{deepdkd2025,
title = {Non-invasive biopsy diagnosis of diabetic kidney disease via deep learning applied to retinal images: a population-based study},
journal = {The Lancet Digital Health},
pages = {100868},
year = {2025},
issn = {2589-7500},
doi = {https://doi.org/10.1016/j.landig.2025.02.008},
url = {https://www.sciencedirect.com/science/article/pii/S2589750025000408},
author = {Ziyao Meng, Zhouyu Guan, Shujie Yu, et al.}
}
```
