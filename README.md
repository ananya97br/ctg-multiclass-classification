# Multiclass Classification of Fetal Health using Cardiotocography (CTG) Data

TEAM #24:
1. ANANYA BELIMALLUR RAJASHEKAR (PES2UG24CS057)
2. AKSHATHA P (PES2UG24CS048)

---

A machine learning reimplementation and comparative evaluation of multiclass fetal health classification using cardiotocography (CTG) data.

This project reproduces the machine learning workflow described in the reference study using the UCI Cardiotocography dataset. The objective is to classify fetal health into three categories — **Normal**, **Suspect**, and **Pathological** and compare the performance of multiple machine learning algorithms with the results reported in the reference paper.

---

## Project Overview

Cardiotocography (CTG) is used to monitor fetal heart rate and uterine contractions during pregnancy. Machine learning techniques can be applied to CTG measurements to assist in identifying different fetal health states.

This project implements a complete machine learning pipeline consisting of:

- Data cleaning and preprocessing
- Exploratory data analysis
- Correlation analysis
- Feature selection using SelectKBest
- Train-test splitting
- Feature standardization
- Training of 10 machine learning models
- Model evaluation
- Comparison with results reported in the reference study

---

## Dataset

The project uses the **Cardiotocography dataset** from the UCI Machine Learning Repository.

The original dataset contains **2,126 observations** derived from cardiotocograms.

### Target Variable

The `fetal_health` target contains three classes:

| Label | Fetal State |
|------:|-------------|
| 1 | Normal |
| 2 | Suspect |
| 3 | Pathological |

After preprocessing and duplicate removal, the dataset used in this project contains **2,113 observations**.

The resulting class distribution is:

| Fetal State | Samples |
|-------------|--------:|
| Normal | 1,646 |
| Suspect | 292 |
| Pathological | 175 |
| **Total** | **2,113** |

Dataset source:

[UCI Cardiotocography Dataset](https://archive.ics.uci.edu/dataset/193/cardiotocography)

---

## Data Preprocessing

The preprocessing workflow is implemented in:

`experiments/01_data_preprocessing.ipynb`

The original dataset contains **2,126 samples**.

The preprocessing stage includes:

1. Loading and inspecting the dataset
2. Checking dataset dimensions and feature types
3. Checking for missing values
4. Identifying duplicate observations
5. Removing duplicate records
6. Verifying the target class distribution
7. Exporting the cleaned dataset

After duplicate removal:

```text
Original samples : 2,126
Duplicates removed: 13
Final samples    : 2,113
```

The cleaned dataset is stored as:

```text
data/processed/fetal_health_cleaned.csv
```

---

## Exploratory Data Analysis and Feature Selection

Exploratory analysis and feature selection are implemented in:

`experiments/02_eda_feature_selection_train_test_split.ipynb`

The notebook includes:

- Fetal health class distribution
- Feature correlation matrix
- Feature importance analysis
- SelectKBest feature selection
- Selection of the 10 highest-scoring features

### SelectKBest

Feature selection is performed using `SelectKBest` with the ANOVA F-test (`f_classif`).

This reduces the original feature space to the **10 most informative features** before model training.

---

## Train-Test Split and Standardization

The cleaned dataset is divided into:

- **70% training data**
- **30% testing data**

To prevent data leakage, standardization is performed **after** the train-test split.

`StandardScaler` is fitted exclusively on the training data.

The processed datasets are exported as:

```text
data/processed/fetal_health_train_processed.csv
data/processed/fetal_health_test_processed.csv
```

---

## Machine Learning Models

A total of **10 machine learning models** are evaluated.

Implemented in:

`experiments/03_mlp_gradient_boosting_xgboost_lightgbm_linear_svm.ipynb`

The models are:

- Multi-Layer Perceptron (MLP)
- Gradient Boosting
- XGBoost
- LightGBM
- Linear Support Vector Machine

Implemented in:

`experiments/04_logistic_regression_knn_decision_tree_random_forest_svm.ipynb`

The models are:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)

---

## Model Evaluation

Models are evaluated using classification metrics including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Because the dataset is imbalanced, the class-specific precision, recall, and F1-score provide additional information beyond overall classification accuracy.

---

## Current Reimplementation Results

The following results were obtained for the 10 models implemented in this project.

| Model | Reimplementation Accuracy |
|-------|--------------------------:|
| LightGBM | 95.90% |
| XGBoost | 95.58% |
| Gradient Boosting | 95.58% |
| Random Forest | 95.27% |
| Multi-Layer Perceptron | 94.95% |
| K-Nearest Neighbors | 93.85% |
| Decision Tree | 93.22% |
| Support Vector Classifier | 91.96% |
| Logistic Regression | 89.59% |
| Linear Support Vector | 89.27% |

---

## Comparison with Reference Results

The reproduced model performance is compared with the accuracy values reported in the reference paper.

| Model | Reimplementation | Reference | Difference |
|-------|-----------------:|----------:|-----------:|
| LightGBM | 95.90% | 95.58% | +0.32 pp |
| XGBoost | 95.58% | 94.95% | +0.63 pp |
| Gradient Boosting | 95.58% | 95.58% | 0.00 pp |
| Random Forest | 95.27% | 94.95% | +0.32 pp |
| Multi-Layer Perceptron | 94.95% | 93.53% | +1.42 pp |
| K-Nearest Neighbors | 93.85% | 94.01% | -0.16 pp |
| Decision Tree | 93.22% | 93.38% | -0.16 pp |
| Support Vector Classifier | 91.96% | 92.11% | -0.16 pp |
| Logistic Regression | 89.59% | 89.59% | 0.00 pp |
| Linear Support Vector | 89.27% | 89.27% | 0.00 pp |

The reproduced results are close to the values reported in the reference paper across all 10 classifiers. Several models reproduce the reported accuracy almost exactly, while the remaining differences are small.

Small differences can arise from implementation details such as random train-test splitting, random seeds, library versions, and model hyperparameters. Not all implementation details required for exact numerical reproduction are necessarily specified in the reference paper.

pp denotes percentage points.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ananya97br/ctg-multiclass-classification.git
cd ctg-multiclass-classification
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Requirements

The main Python libraries used are:

```text
pandas
numpy
matplotlib
scikit-learn
xgboost
lightgbm
ucimlrepo
joblib
```

The experiments can be executed using Google Colab.

---

## Running the Project

For reproducibility, execute the notebooks in the following order:

```text
1. experiments/01_data_preprocessing.ipynb

2. experiments/02_eda_feature_selection.ipynb

3. experiments/03_mlp_gradient_boosting_xgboost_lightgbm_linear_svm.ipynb
                         and
   experiments/04_logistic_regression_knn_decision_tree_random_svm.ipynb
```

The two model-training notebooks can be executed independently after the preprocessing and feature-selection stages have generated the processed train and test datasets.

---

## Reproducibility

- A fixed train-test split is used.
- Random states are specified where supported.
- The same selected features are used across models.
- All models use the same training and testing datasets.
- Standardization parameters are derived exclusively from the training set.
- Processed datasets are stored separately from the raw data.

---

## Contributors

### Contributor 1 - ANANYA BELIMALLUR RAJASHEKAR

- Multi-Layer Perceptron
- Gradient Boosting
- XGBoost
- LightGBM
- Linear Support Vector Machine
- Model evaluation and comparison

### Contributor 2 - AKSHATHA P

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Classifier
- Model evaluation and comparison

---
