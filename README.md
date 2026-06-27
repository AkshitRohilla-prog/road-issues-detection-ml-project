# Road Issues Detection using Deep Learning

This project presents an explainable deep learning pipeline for road damage and surface defect classification in a smart-city monitoring context.

## Project Overview
The goal of this project is to classify road-surface images into three categories:

- Pothole Issues
- Damaged Road issues
- Mixed Issues

The project was implemented in Kaggle Notebook using PyTorch.

## Models Used
The following models were developed and compared:

- Custom CNN Baseline
- EfficientNetV2-S
- ConvNeXt-Tiny

## Explainability
To improve interpretability, the project includes:

- Grad-CAM
- SHAP

## Best Model
Based on macro F1-score and overall comparison, the best-performing model was:

- EfficientNetV2-S

## Repository Structure
- `Codes/` → executed Kaggle notebook
- `Results/` → saved outputs including metrics, figures, models, and histories
- `Datasets/` → dataset source link
- `Frontend/` → deployment files
- `Report/` → Overleaf/report/presentation links

## Dataset
Road Issues Detection Dataset on Kaggle:
https://www.kaggle.com/datasets/programmerrdai/road-issues-detection-dataset

## Main Outputs
The final project includes:

- trained `.pth` model files
- CSV metrics and comparisons
- confusion matrices
- training curves
- Grad-CAM explanations
- SHAP explanations
- error analysis outputs

## Author
Akshit Rohilla
