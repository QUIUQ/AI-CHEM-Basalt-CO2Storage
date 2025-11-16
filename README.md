# AI-CHEM-Basalt-CO2Storage
1. Dataset
Folder: base_model/

This folder contains the base model built on TOUGHREACT, which serves as the foundation for generating simulation data.

Folder: run_data_code/

This folder contains scripts for producing the full dataset.

gene0: The main script used to generate the complete dataset from the base model setup.

2. DL_Model
Folder: Model/

This folder includes:

Deep Learning model implementations

Trained model checkpoints for different architectures

Saved paths for each trained model

Training Script

train555-MSE.py
This script trains the DL models using the MSE loss function.

Post-processing Script

post555.py
This script handles:

Visualization of predictions and results

Dataset post-processing (e.g., normalization, reshaping, plotting)
