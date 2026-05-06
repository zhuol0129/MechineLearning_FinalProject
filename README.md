# Decoding Risk Factors of Alzheimer's Disease

Classifying Alzheimer's disease using demographic, health, lifestyle, and cognitive variables.
Pipeline covers preprocessing, exploratory data analysis (EDA), dimensionality reduction (LLE),
and model evaluation (Decision Tree / SVM).

---

## Files Required

Place the following files in the same folder before running:

- `alzheimers_disease_data.csv` — source dataset
- `preprocessing.ipynb` — main analysis notebook

---

## Setup

**1. Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

**2. Update the working directory**

In the first code cell of `preprocessing.ipynb`, change the path to match your local folder:

```python
os.chdir("/your/project/folder")
```

---

## Pipeline Overview

### 1. Preprocessing

- Drops non-informative identifier columns (`PatientID`, `DoctorInCharge`)
- One-hot encodes `Ethnicity` into four binary columns:
  `Ethnicity_Caucasian`, `Ethnicity_AfricanAmerican`, `Ethnicity_Asian`, `Ethnicity_Other`
- Creates two analysis datasets:
  - **Full dataset** (`df_full`): all variables retained
  - **Risk-only dataset** (`df_risk`): symptom variables removed

Symptom variables excluded from `df_risk`:
`MMSE`, `FunctionalAssessment`, `MemoryComplaints`, `BehavioralProblems`,
`ADL`, `Confusion`, `Disorientation`, `PersonalityChanges`,
`DifficultyCompletingTasks`, `Forgetfulness`

---

### 2. Exploratory Data Analysis (EDA)

- Descriptive statistics (mean, SD, min, max) for all continuous variables
- Histograms with KDE overlays for continuous variables
- KDE distribution plots split by Diagnosis group (Healthy vs. Alzheimer's)
- Discrimination gap bar charts for binary and categorical predictors
- Outlier detection using the IQR rule (1.5 × IQR)
- Correlation heatmaps (full and sorted by absolute correlation strength)
- Per-variable correlation with `Diagnosis`, ranked highest to lowest

---

### 3. Dimensionality Reduction (LLE)

Locally Linear Embedding applied with:
- `n_components = 5`
- `n_neighbors = 15`
- `method = "standard"`

Standard scaling is applied to continuous columns before LLE.
LLE is fit on training data only; the learned transform is applied to the test set.

---

### 4. Model Training and Evaluation

Each model is run under **12 conditions** (2 datasets x 2 feature sets x 3 models):

| Dataset   | Feature Set       | Model         |
|-----------|-------------------|---------------|
| Full      | No LLE (raw)      | Decision Tree |
| Full      | No LLE (scaled)   | SVM Linear    |
| Full      | No LLE (scaled)   | SVM RBF       |
| Full      | LLE               | Decision Tree |
| Full      | LLE               | SVM Linear    |
| Full      | LLE               | SVM RBF       |
| Risk-only | No LLE (raw)      | Decision Tree |
| Risk-only | No LLE (scaled)   | SVM Linear    |
| Risk-only | No LLE (scaled)   | SVM RBF       |
| Risk-only | LLE               | Decision Tree |
| Risk-only | LLE               | SVM Linear    |
| Risk-only | LLE               | SVM RBF       |

**Decision Tree** — hyperparameters tuned via 5-fold cross-validated grid search (scored by ROC-AUC):
- `max_depth`: [2, 3, 5]
- `min_samples_leaf`: [5, 10, 20]
- `class_weight`: [None, "balanced"]

**SVM** — `class_weight="balanced"`, `random_state=42`

---

### 5. Evaluation Metrics

All models report:
- ROC-AUC
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- ROC curve

Decision Tree models additionally produce:
- Top-15 feature importance bar chart (Gini impurity)
- Full tree diagram

---

## Reproducibility

| Setting           | Value                                          |
|-------------------|------------------------------------------------|
| Train / test split | 70% / 30%                                    |
| `random_state`    | 42                                             |
| `stratify`        | `y` (class proportions preserved across splits)|
