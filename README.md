# DeepDKD - Official Implementation

## Overview
Welcome to the official GitHub repository for DeepDKD, the implementation of the research paper titled "Screening and Non-invasive Biopsy Diagnosis of Diabetic Kidney Disease Using Deep Learning Applied to Retinal Fundus Images." 
This project provides the codebase for training models for the detection of Diabetic Kidney Disease (DKD) and differentiating isolated diabetic nephropathy (DN) from non-diabetic kidney disease (NDKD) using deep learning techniques.

## Project Configuration
To set up the project environment, please refer to the `requirements.txt` file. You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Pretraining
The DeepDKD system used weakly supervised momentum contrastive learning as a method for extracting transferable visual representations of retinal fundus images with further supervised training in two tasks (detecting DKD and differentiating isolated from NDKD). 
To initiate the pretraining of the DeepDKD system, execute the pretraining script:

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

