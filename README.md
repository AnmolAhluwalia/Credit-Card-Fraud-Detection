# Credit Card Fraud Detection

## Overview
This repository contains a comprehensive Jupyter notebook (`credit-card-fraud-detection.ipynb`) that explores credit card fraud detection using machine learning. The project covers everything from exploratory data analysis (EDA) and feature engineering through model training, evaluation, feature importance, and visualization.

## Table of Contents
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
4. [Data Visualization](#data-visualization)  
5. [Data Preprocessing](#data-preprocessing)  
6. [Model Training & Evaluation](#model-training--evaluation)  
7. [Feature Importance](#feature-importance)  
8. [PCA Visualization](#pca-visualization)  
9. [AUC-ROC Curves](#auc-roc-curves)  
10. [Conclusions](#conclusions)  
11. [Usage](#usage)  
12. [Tech Stack](#tech-stack)  
13. [Acknowledgments](#acknowledgments)  
14. [License](#license)  

---

## Introduction
This project aims to detect fraudulent credit card transactions using a variety of supervised learning models. The analysis begins with understanding the dataset, visualizing patterns, handling class imbalance, scaling data, and finally training models like K-Nearest Neighbors, SVM, Decision Tree, Random Forest, XGBoost, and more.

## Dataset
- **Source**: Kaggle Credit Card Fraud dataset (European cardholders, September 2013)  
- **Size**: 284,807 transactions, of which 492 are fraudulent (~0.172%)  
- **Features**:
  - **V1 – V28**: PCA-transformed anonymized features  
  - **Time**: Seconds elapsed since the first transaction  
  - **Amount**: Transaction amount  
  - **Class**: Target variable (0 = legitimate, 1 = fraud) :contentReference[oaicite:0]{index=0}

## Exploratory Data Analysis (EDA)
- Checked for missing values (none found)
- Descriptive statistics for `Time`, `Amount`, and `Class`
- Pearson correlation matrix to detect feature relationships and possible redundancies

## Data Visualization
- Pie chart to show class imbalance between legitimate and fraudulent transactions  
- Heatmap of feature correlations  
- Hourly transaction trends (fraud vs. non-fraud) to observe time-based patterns

## Data Preprocessing
- Extracted `X` (features) and `y` (target)
- Split data into train and test sets (80% train, 20% test)
- Standardized features using `StandardScaler` to normalize scale-sensitive features

## Model Training & Evaluation
- Trained multiple models:  
  - K-Nearest Neighbors  
  - Support Vector Classifier  
  - Gaussian Naive Bayes  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
  - LightGBM  
  - Gradient Boosting Classifier  
  - AdaBoost  
  - Logistic Regression  
- Evaluated using accuracy, F1-score, MAE, MSE, RMSE, R² score, classification reports, and confusion matrices plotted via Plotly

## Feature Importance
- Extracted and visualized feature importances for applicable models (excluding SVC and Logistic Regression)

## PCA Visualization
- Reduced the feature space to 2 principal components using PCA  
- Plotted a scatterplot with fraud vs. non-fraud points to visualize separation

## AUC-ROC Curves
- Generated ROC curves and computed AUC scores for:
  - KNN (0.9285)
  - Random Forest (0.9526)
  - XGBoost (0.9766)
  - Gradient Boosting (0.7855)
  - Decision Tree (0.8926)
  - Logistic Regression (0.9751)  
- **Best performer**: XGBoost  
  - **AUC**: 0.9766  
  - **Accuracy**: ~0.9996

## Conclusions
- XGBoost yielded the best performance considering AUC and accuracy.  
- Important insights were gained on feature relevance, model comparison, and dataset characteristics.

## Usage
To run this project locally:
```bash
git clone https://github.com/AnmolAhluwalia/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
# (Optional) create and activate a virtual environment
pip install -r requirements.txt
jupyter notebook
# Open credit-card-fraud-detection.ipynb and follow each section
