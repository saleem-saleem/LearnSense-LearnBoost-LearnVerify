#  LearnSense–LearnBoost–LearnVerify

### A Three-Stage Framework for Confidence-Aware Student Performance Prediction

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Educational%20AI-success.svg)]()
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble%20Learning-orange.svg)]()

---

##  Overview

**LearnSense–LearnBoost–LearnVerify** is a novel three-stage Artificial Intelligence (AI) framework designed to improve **student performance prediction**, **learning analytics**, and **confidence-aware educational decision-making**.

The framework integrates three complementary components:

| Stage | Framework | Algorithm | Purpose |
|---------|-----------|------------|----------|
| Stage 1 | **LearnSense** | **TG-PSF** | Target-guided feature selection |
| Stage 2 | **LearnBoost** | **RGEAB-TE** | Diversity-constrained boosted ensemble learning |
| Stage 3 | **LearnVerify** | **CCSG-PR** | Confidence-calibrated selective prediction |

Unlike conventional approaches that independently perform feature selection, prediction, and validation, LearnSense–LearnBoost–LearnVerify combines these tasks into a unified pipeline to improve:

- ✅ Prediction Accuracy
- ✅ Feature Relevance
- ✅ Model Robustness
- ✅ Confidence Calibration
- ✅ Reliability of Educational Decisions

---

#  Key Features

###  LearnSense (TG-PSF)

**Target-Guided Projection and Similarity Filtering**

- Target-oriented feature relevance estimation
- Similarity-based feature redundancy removal
- Adaptive feature subset optimization
- Noise reduction and dimensionality minimization
- Improved downstream learning performance

###  LearnBoost (RGEAB-TE)

**Regression-Guided Error-Adaptive Boosted Tree Ensemble**

- Error-adaptive boosting strategy
- Diversity-aware ensemble construction
- Confidence-weighted learner aggregation
- Regression-guided instance reweighting
- Enhanced predictive generalization

###  LearnVerify (CCSG-PR)

**Confidence-Calibrated Similarity-Guided Prediction Refinement**

- Confidence estimation and calibration
- Similarity-aware prediction validation
- Selective prediction acceptance/rejection
- Reliability-guided prediction refinement
- Uncertainty-aware educational analytics

###  Framework Advantages

- End-to-end learning pipeline
- Confidence-aware predictions
- Reduced feature redundancy
- Robust ensemble learning
- Improved reliability and interpretability
- Scalable to heterogeneous educational datasets

---

#  Applications

The proposed framework can be applied to a wide range of educational analytics tasks:

### Student Performance Prediction
Forecast academic outcomes using demographic, behavioral, and academic attributes.

###  Early Risk Identification
Detect students at risk of poor academic performance, dropout, or failure.

###  Personalized Learning Systems
Support adaptive and personalized educational interventions.

###  Learning Analytics
Analyze learning patterns, engagement behavior, and educational outcomes.

###  Educational Decision Support
Assist educators and administrators in data-driven academic planning.

###  Intelligent Tutoring Systems
Enable confidence-aware recommendations and adaptive feedback.

### Educational Data Mining Research
Provide a benchmark framework for educational prediction studies.

---

#  Repository Structure

```text
learnsense-learnboost-learnverify/
├── src/
│   ├── tgpsf.py
│   ├── rgeabte.py
│   ├── ccsgpr.py
│   ├── pipeline.py
│   └── preprocessing.py
│
├── baselines/
│   └── baselines.py
│
├── evaluation/
│   ├── metrics.py
│   └── calibration.py
│
├── figures/
│   ├── reliability_separate.py
│   └── risk_coverage_curves.py
│
├── scripts/
│   └── train_evaluate.py
│
├── config/
│   ├── tgpsf_config.yaml
│   ├── rgeabte_config.yaml
│   └── ccsgpr_config.yaml
│
├── data/
│   └── splits/
│
├── requirements.txt
└── README.md
```

---

#  Framework Modules

## 1️LearnSense (TG-PSF)

**File:** `src/tgpsf.py`

### Responsibilities

- Feature relevance computation
- Target-guided projection analysis
- Similarity-based feature filtering
- Adaptive threshold selection
- Feature reduction

### Input

```text
X : Feature Matrix
y : Target Labels
```

### Output

```text
Optimized Feature Subset
```

### Main Functions

```python
fit()
transform()
fit_transform()
reduction_pct()
```

---

## LearnBoost (RGEAB-TE)

**File:** `src/rgeabte.py`

### Responsibilities

- Ensemble learner construction
- Error-adaptive boosting
- Confidence estimation
- Diversity-aware learner selection
- Prediction aggregation

### Input

```text
Optimized Features from TG-PSF
```

### Output

```text
Prediction Labels
Prediction Probabilities
```

### Main Functions

```python
fit()
predict()
predict_proba()
```

---

##  LearnVerify (CCSG-PR)

**File:** `src/ccsgpr.py`

### Responsibilities

- Confidence calibration
- Similarity-guided refinement
- Reliability estimation
- Prediction acceptance/rejection

### Input

```text
Predictions
Confidence Scores
Feature Space Similarity
```

### Output

```text
Refined Predictions
Confidence Values
Coverage Decisions
```

### Main Functions

```python
fit()
predict()
```

---

## Unified Pipeline

**File:** `src/pipeline.py`

Integrates all three stages into a single end-to-end framework.

### Main Class

```python
LearnSenseBoostVerify
```

### Pipeline Flow

```text
Dataset
   │
   ▼
TG-PSF
   │
   ▼
RGEAB-TE
   │
   ▼
CCSG-PR
   │
   ▼
Final Prediction
```

---

##  Evaluation Module

**Directory:** `evaluation/`

### metrics.py

Computes:

- Accuracy
- Precision
- Recall
- F1-score
- RMSE
- Paired t-test

### calibration.py

Computes:

- Expected Calibration Error (ECE)
- Brier Score
- Reliability Statistics

---

##  Visualization Module

**Directory:** `figures/`

### reliability_separate.py

Generates:

- Reliability Diagrams
- Confidence Histograms

### risk_coverage_curves.py

Generates:

- Risk-Coverage Curves
- Selective Prediction Analysis

---

#  Datasets

The framework was evaluated using five public educational datasets.

| ID | Dataset |
|------|----------|
| D1 | Student Performance - Multiple Linear Regression |
| D2 | Student Performance Factors |
| D3 | Student Performance (UCI #320) |
| D4 | Student Performance on an Entrance Examination |
| D5 | Predict Students Dropout and Academic Success |

All datasets are publicly available through Kaggle and UCI Machine Learning Repository.

---

# 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/[USERNAME]/learnsense-learnboost-learnverify.git

cd learnsense-learnboost-learnverify
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install TensorFlow for deep learning baselines:

```bash
pip install tensorflow>=2.12.0
```

---

# Required Python Packages

```text
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
pyyaml
joblib
xgboost
lightgbm
tensorflow
keras
```

---

#  Quick Start

```python
from src.pipeline import LearnSenseBoostVerify

model = LearnSenseBoostVerify()

model.fit(X_train, y_train)

accuracy = model.score_full_coverage(X_test, y_test)

print("Accuracy:", accuracy)
```

---

#  Running Individual Modules

## Run LearnSense

```python
from src.tgpsf import TGPSF

selector = TGPSF()

X_selected = selector.fit_transform(X, y)
```

---

## Run LearnBoost

```python
from src.rgeabte import RGEABTE

ensemble = RGEABTE()

ensemble.fit(X_train, y_train)

predictions = ensemble.predict(X_test)
```

---

## Run LearnVerify

```python
from src.ccsgpr import CCSGPR

validator = CCSGPR()

labels, confidence, accepted = validator.predict(
    X_test,
    probabilities
)
```

---

#  Reproducing Paper Results

```bash
python scripts/train_evaluate.py \
    --dataset D1 \
    --csv_path data/D1.csv \
    --target Performance_Index
```

The script automatically:

- Loads the dataset
- Uses predefined seed splits
- Performs TG-PSF feature optimization
- Trains RGEAB-TE ensemble
- Applies CCSG-PR refinement
- Computes evaluation metrics
- Saves experimental results

### Evaluation Seeds

```text
42
77
101
202
555
```

---

# Generated Outputs

The framework produces:

### Feature Optimization Results

```text
Selected Features
Reduction Percentage
Feature Importance
```

### Prediction Results

```text
Predicted Labels
Prediction Probabilities
```

### Confidence Analysis

```text
Confidence Scores
Coverage Rate
Accepted Instances
Rejected Instances
```

### Evaluation Metrics

```text
Accuracy
Precision
Recall
F1 Score
RMSE
ECE
Brier Score
```




**LearnSense–LearnBoost–LearnVerify: Improving educational predictions through feature intelligence, adaptive ensemble learning, and confidence-aware decision making.**
