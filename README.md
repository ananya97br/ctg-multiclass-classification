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
- Additional soft voting ensemble using LightGBM, XGBoost, and CatBoost
- Further tuning, imbalance-handling, feature-set, voting, and stacking experiments
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

### Additional Ensemble Experiment

Implemented in:

`experiments/05_soft_voting_ensemble.ipynb`

The additional experiment combines:

- LightGBM
- XGBoost
- CatBoost

The three models are combined using **equal-weight soft voting**. Their predicted class probabilities are averaged, and the class with the highest average probability is selected as the final prediction.

Unlike the original 10-model experiments, this ensemble uses **all 21 CTG input features** rather than the 10 features selected using SelectKBest. Because LightGBM, XGBoost, and CatBoost are tree-based models, feature standardization was not required for this experiment.

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

The following results were obtained for the 10 reimplemented models and the additional soft voting ensemble.

| Model | Accuracy |
|-------|---------:|
| **Soft Voting Ensemble (21 features)** | **96.06%** |
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

For the soft voting ensemble, the additional imbalance-aware metrics were:

| Metric | Score |
|--------|------:|
| Accuracy | **96.06%** |
| Balanced Accuracy | **91.82%** |
| Macro F1-score | **0.9377** |

The ensemble achieved the highest observed test accuracy in the project. It also achieved strong balanced accuracy and macro F1-score, although the Suspect class remained more difficult to classify than the Normal and Pathological classes.

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

## Additional Soft Voting Ensemble Result

The soft voting ensemble is an **additional experiment beyond the reference study** and is therefore not included in the paper-to-reimplementation comparison table above.

The ensemble combines LightGBM, XGBoost, and CatBoost using equal weights:

```text
LightGBM + XGBoost + CatBoost
            ↓
  Average class probabilities
            ↓
      Final prediction
```

Using all 21 CTG features, the ensemble achieved **96.06% accuracy**, which is slightly higher than the best individual reimplementation result of **95.90%** from LightGBM.

---

## Further Experiments

Several follow-up experiments were conducted to test hyperparameter tuning, class-imbalance handling, use of all available features, and additional ensemble strategies. Unless otherwise noted, these experiments use the **10 features selected with SelectKBest**.

For comparison, the main reimplementation baselines were **KNN: 93.85%**, **SVC: 91.96%**, **Random Forest: 95.27%**, and the best individual model, **LightGBM: 95.90%**.

The boosted-model soft-voting run averages the class probabilities produced by **LightGBM, XGBoost, and CatBoost** using equal weights. It uses **all 21 CTG input features**, does **not** apply feature scaling, and uses a new **stratified 70/30 train-test split** with `random_state=42`. This run achieved approximately **96.06% accuracy**.

| Experiment | Accuracy |
|------------|---------:|
| KNN, grid search (`k=3`, distance weights, Manhattan distance) | 94.01% |
| SVM (RBF), grid search (`C=100`) | 95.11% |
| Random Forest, 300 trees, fixed seed | 95.58% |
| Random Forest, balanced class weights | 95.27% |
| Random Forest, SMOTE oversampling | 94.95% |
| Random Forest, all 21 features | 94.95% |
| Random Forest, random search with cross-validation | 95.11% |
| Soft voting: Logistic Regression + KNN + Random Forest + SVM | 95.11% |
| Stacking: same four models, Logistic Regression meta-model | 95.27% |
| **Soft voting: LightGBM + XGBoost + CatBoost, all 21 features** | **96.06%** |

Among these follow-up experiments, the 300-tree Random Forest improved to **95.58%**, while balanced class weights and SMOTE did not improve on the original Random Forest baseline. The boosted-model soft-voting ensemble remained the highest-accuracy result among the experiments listed here.

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
catboost
ucimlrepo
joblib
```

The experiments can be executed using Google Colab.

---

## Running the Project

For reproducibility, execute the notebooks in the following order:

```text
1. experiments/01_data_preprocessing.ipynb

2. experiments/02_eda_feature_selection_train_test_split.ipynb

3. experiments/03_mlp_gradient_boosting_xgboost_lightgbm_linear_svm.ipynb

4. experiments/04_logistic_regression_knn_decision_tree_random_forest_svm.ipynb

5. experiments/05_soft_voting_ensemble.ipynb
```

The two original model-training notebooks can be executed independently after the preprocessing and feature-selection stages have generated the processed train and test datasets.

The soft voting ensemble notebook is an additional experiment. It uses `fetal_health_cleaned.csv` so that all 21 CTG input features are available.

---

## Reproducibility

- Fixed random states are specified where supported.
- The 10 original reimplemented models use the same processed training and testing datasets.
- The original model experiments use the 10 features selected with SelectKBest.
- Standardization parameters for the original experiments are derived exclusively from the training set.
- The soft voting extension uses all 21 CTG features with a 70/30 stratified split and `random_state=42`.
- LightGBM, XGBoost, and CatBoost are not standardized in the ensemble experiment because they are tree-based models.
- Processed datasets are stored separately from the raw data.

---
