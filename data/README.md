# Dataset

This directory contains the cardiotocography (CTG) datasets used in the reimplementation of the fetal health classification study.

The project is based on the Cardiotocography dataset associated with the work of Ayres-de-Campos et al. and distributed through the UCI Machine Learning Repository.


## Directory Structure

```text
data/
├── raw/
│   └── CTG.xls
│
└── processed/
    ├── fetal_health_cleaned.csv
    ├── fetal_health_train_processed.csv
    └── fetal_health_test_processed.csv
```

## Data Source - raw/CTG.xls

The underlying data originates from the **Cardiotocography dataset** available through the **UCI Machine Learning Repository**.

The dataset consists of measurements extracted from fetal cardiotocograms, including fetal heart rate (FHR) and uterine contraction (UC) characteristics.

A total of **2,126 cardiotocograms** were automatically processed to extract diagnostic features. The recordings were also classified by three expert obstetricians, with a consensus classification assigned to each observation.

The dataset contains:

- **2,126 observations**
- **21 input features**
- a 10-class morphological classification (`CLASS`)
- a 3-class fetal-state classification (`NSP`)

For this project, the three-class fetal-state classification is used.

1. Normal
2. Suspect
3. Pathological

### Dataset page:  
[UCI Cardiotocography Dataset](https://archive.ics.uci.edu/dataset/193/cardiotocography) 

## Processed Data

### processed/fetal_health_cleaned.csv

Cleaned dataset produced during the preprocessing stage.

- Original observations: 2,126
- Duplicate observations removed: 13
- Final observations: 2,113
- Input features: 21
- Target: fetal_health


The final target distribution is:

| Class | Number of Samples |
|-------|------------------:|
| Normal | 1,646 |
| Suspect | 292 |
| Pathological | 175 |
| **Total** | **2,113** |

These values reproduce the preprocessing counts reported in the reference paper.
This cleaned dataset is used as the input for exploratory data analysis and feature selection.

### processed/fetal_health_train_processed.csv

Training dataset produced after feature selection, a 70/30 train-test
split, and standardization.
Contains the 10 selected features and the fetal health target.

### processed/fetal_health_test_processed.csv

Testing dataset produced from the same 70/30 split and transformed using
the StandardScaler fitted on the training data.