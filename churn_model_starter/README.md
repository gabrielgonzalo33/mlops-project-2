# Churn Model Starter Files

This folder contains the starter files for the **Create a Feature Branch and Push ML Artifacts** HoL activity.

## Folder Structure

```
churn_model_starter/
├── notebooks/
│   └── churn_model.ipynb      # Baseline churn prediction notebook
├── preprocessing/
│   └── clean_data.py          # Data preprocessing utilities
├── configs/
│   └── churn_config.yaml      # Model configuration parameters
└── README.md                  # This file
```

## Files Description

### notebooks/churn_model.ipynb
A Jupyter notebook containing the baseline churn prediction workflow:
- Data loading and exploration
- Preprocessing pipeline
- Model training (Random Forest)
- Evaluation metrics

### preprocessing/clean_data.py
Python module with preprocessing functions:
- `load_data()` - Load CSV data
- `handle_missing_values()` - Handle missing data with configurable strategies
- `encode_categorical()` - One-hot encode categorical features
- `scale_features()` - Standardize or normalize numeric features

### configs/churn_config.yaml
YAML configuration file containing:
- Preprocessing parameters
- Model hyperparameters
- Feature definitions
- Training settings

## Instructions

1. Download and extract these files to your local Git repository
2. Follow the HoL activity steps to create a feature branch and commit these artifacts
3. Remember to clear notebook outputs before committing

## Note

Before committing the notebook, ensure you:
- Clear all cell outputs (Cell → All Output → Clear)
- Restart kernel and run all cells to verify the notebook runs correctly
