# AI-CHEM-Basalt-CO2Storage
## 1. base_model

This folder contains the base model built on TOUGHREACT, which serves as the foundation for generating simulation data.

## 2. Folder: run_data_code      

This folder contains scripts for producing the full dataset.

gene0: The main script used to generate the complete dataset from the base model setup.

## 3. Model

This folder includes:

Deep Learning model 

train555-MSE.py:This script trains the DL models using the MSE loss function.

post555.py:This script handles Visualization of predictions and results AND Dataset post-processing (e.g., normalization, reshaping, plotting)
