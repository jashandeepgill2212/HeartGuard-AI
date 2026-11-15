# HeartGuard AI – Intelligent Heart Disease Prediction & Risk Analysis System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.51.0-brightgreen.svg)](https://shap.readthedocs.io/)
[![ReportLab](https://img.shields.io/badge/ReportLab-5.1.0-blueviolet.svg)](https://www.reportlab.com/)

---

## 📌 Executive Summary

**HeartGuard AI** is an advanced clinical decision support prototype and machine learning analytics platform designed to analyze patient cardiovascular parameters and estimate the probability of **Coronary Artery Disease (CAD)**. 

Built from the ground up for a final-year B.Tech Capstone Project, the system combines **7 tuned machine learning algorithms**, leak-proof Scikit-Learn preprocessing pipelines, **Explainable AI via SHAP (SHapley Additive exPlanations)**, interactive population epidemiology analytics, and automated downloadable clinical PDF documentation.

> **Academic & Clinical Disclaimer:** HeartGuard AI is an academic research and educational decision-support prototype. It is **not** an FDA-cleared medical diagnostic device. Under no circumstances should its outputs replace evaluation, laboratory diagnostics, coronary catheterization, or the clinical judgment of a licensed cardiologist.

---

## 🌟 Key System Capabilities

1. **Multi-Model Machine Learning Engine:**
   - Implements, tunes, and evaluates 7 distinct classifiers:
     1. Logistic Regression
     2. Random Forest *(Champion Model: 90.16% Accuracy, 96.43% Recall, 0.9535 ROC-AUC)*
     3. Decision Tree
     4. Support Vector Machine (RBF/Linear kernels with calibrated probability)
     5. K-Nearest Neighbors
     6. Gradient Boosting
     7. Extreme Gradient Boosting (XGBoost)
   - Evaluated on a holdout test set with **Stratified 5-Fold Cross-Validation** to guarantee generalizability.

2. **Leak-Proof Preprocessing Architecture:**
   - Encapsulated inside Scikit-Learn `ColumnTransformer` and `Pipeline` objects.
   - Continuous attributes: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
   - Discrete attributes: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.
   - Transformers fit strictly on training folds to prevent data leakage.

3. **Advanced Explainable AI (SHAP):**
   - **Global Cohort Importance:** Visualizes overall model drivers (mean $|\\text{SHAP}|$ across the cohort).
   - **Local Patient Waterfall Decomposition:** Deconstructs an individual patient's prediction into positive risk accelerators (red) vs protective buffers (teal).
   - **Causality Guidance:** Clearly distinguishes statistical pattern attribution from biological medical causation.

4. **Interactive Clinical Dashboard:**
   - Custom medical technology design system styled with dark navy (`#0B192C`), teal (`#0D9488`), and coral/crimson (`#E63946`) accents.
   - One-click clinical testing presets: *Healthy Low-Risk*, *Moderate-Risk Hypertensive*, *High-Risk Ischemic*, and *Severe Multi-Vessel CAD Geriatric Patient*.
   - Dynamic probability gauge and AHA/ACC aligned risk categories (*Low*, *Borderline*, *Elevated*, *High/Critical*).

5. **Automated Clinical PDF Report Generation:**
   - Instant downloadable PDF clinical summaries built with ReportLab, featuring patient demographics, biomarker tables with reference ranges, SHAP attribution summaries, and evidence-based guideline recommendations (AHA/ACC / ESC).

6. **Batch Patient Screening & Triage:**
   - High-throughput CSV batch intake with instant population risk stratification and exportable registry CSVs.

---

## 📊 Dataset Profile & Clinical Target Conversion

The system utilizes the gold-standard **UCI Cleveland Heart Disease Database** collected by Dr. Robert Detrano at the Cleveland Clinic Foundation:

| Metric | Detail |
|---|---|
| **Original Source** | UCI Machine Learning Repository / Cleveland Clinic Foundation (Detrano et al., 1989) |
| **Total Cohort Records** | 303 patient records |
| **Clinical Features** | 13 physiological parameters (5 continuous, 8 categorical/ordinal) |
| **Duplicate Records** | 0 duplicates |
| **Missing Values** | 6 total cells (`ca`: 4 missing, `thal`: 2 missing), imputed strictly via training median/mode |
| **Target Distribution** | 164 Healthy / Low Risk (54.13%) vs. 139 CAD Present (45.87%) |

### Target Binarization Methodology
In the original clinical records, the angiographic coronary narrowing score (`num`) ranged from **0** (diameter narrowing $< 50\\%$) to **1, 2, 3, 4** (diameter narrowing $\\ge 50\\%$ across 1 to 4 epicardial vessels). 
For binary risk classification, value **0** designates **Absence of Significant CAD / Healthy**, while values **1–4** are mapped to **1 (Coronary Artery Disease Indicated)**.

---

## 📈 Empirical Model Benchmark Results

All metrics reflect **genuine model execution** on the 20% holdout test set ($N=61$ patients) with Stratified 5-Fold Cross-Validation:

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | Clinical Composite Score |
|---|---|---|---|---|---|---|---|
| **Random Forest (Champion)** | **90.16%** | **84.38%** | **96.43%** | **84.85%** | **0.9000** | **0.9535** | **0.9439** |
| **K-Nearest Neighbors** | 88.52% | 86.21% | 89.29% | 87.88% | 0.8772 | 0.9621 | 0.9166 |
| **Logistic Regression** | 86.89% | 83.33% | 89.29% | 84.85% | 0.8621 | 0.9632 | 0.9133 |
| **XGBoost** | 86.89% | 85.71% | 85.71% | 87.88% | 0.8571 | 0.9421 | 0.8911 |
| **Gradient Boosting** | 86.89% | 85.71% | 85.71% | 87.88% | 0.8571 | 0.9280 | 0.8855 |
| **Support Vector Machine** | 83.61% | 80.00% | 85.71% | 81.82% | 0.8276 | 0.9426 | 0.8839 |
| **Decision Tree** | 80.33% | 76.67% | 82.14% | 78.79% | 0.7931 | 0.8339 | 0.8193 |

### Clinical Selection Strategy
In cardiovascular screening triage, a **False Negative** (discharging a patient with occult CAD) has lethal implications (unheralded myocardial infarction or sudden cardiac arrest). In contrast, a **False Positive** is safely managed with non-invasive outpatient diagnostics.
Therefore, the model selection strategy optimizes a composite index:
$$\\text{Composite Score} = 0.40 \\cdot \\text{ROC-AUC} + 0.35 \\cdot \\text{Recall} + 0.25 \\cdot \\text{F1}$$
**Random Forest** achieved the highest score (**0.9439**), missing only 1 CAD patient in holdout testing (96.43% Sensitivity).

---

## 📁 Repository Directory Structure

```text
heart disease/
│
├── app.py                      # Main Streamlit Web Application Dashboard
├── train.py                    # Standalone ML training, tuning & benchmark runner
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & clinical specifications
│
├── data/
│   ├── raw/
│   │   ├── processed.cleveland.data    # Authentic UCI Cleveland dataset
│   │   ├── processed.hungarian.data    # Multi-center reference
│   │   ├── processed.switzerland.data  # Multi-center reference
│   │   └── processed.va.data           # Multi-center reference
│   └── processed/
│       ├── heart_cleveland_clean.csv   # Audited & cleaned dataset
│       ├── dataset_summary.json        # Descriptive statistics & audit log
│       └── sample_patient_batch.csv    # 15 pre-built patient records for batch testing
│
├── models/
│   └── artifacts/
│       ├── best_model.joblib           # Champion Random Forest pipeline
│       ├── logistic_regression.joblib  # Tuned Logistic Regression
│       ├── random_forest.joblib        # Tuned Random Forest
│       ├── decision_tree.joblib        # Tuned Decision Tree
│       ├── support_vector_machine.joblib # Tuned SVM
│       ├── k_nearest_neighbors.joblib  # Tuned KNN
│       ├── gradient_boosting.joblib    # Tuned Gradient Boosting
│       ├── xgboost.joblib              # Tuned XGBoost
│       ├── preprocessing_pipeline.joblib # Fitted ColumnTransformer
│       ├── model_metrics.json          # Actual test evaluation metrics
│       ├── cv_results.json             # 5-fold cross-validation scores
│       ├── roc_pr_data.joblib          # Raw curve coordinates for all models
│       ├── shap_global_summary.joblib  # Precomputed SHAP feature rankings
│       └── transformed_feature_names.joblib
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Feature schemas, clinical metadata, color palettes
│   ├── data_loader.py          # Data ingestion, validation, and cleaning
│   ├── eda.py                  # Exploratory Data Analysis & Plotly chart builders
│   ├── ml_pipeline.py          # Preprocessing, Stratified GridSearchCV, evaluation
│   ├── xai_shap.py             # SHAP XAI engine (waterfall, summary, causality notes)
│   └── report_generator.py     # PDF clinical documentation engine (ReportLab)
│
├── static/
│   └── style.css               # Premium custom healthcare technology styling
│
└── tests/
    └── test_pipeline.py        # Automated test suite (Data, Model, SHAP, PDF, Batch)
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10 or 3.11 installed.
- Recommended virtual environment:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Automated Tests
```powershell
pytest -v tests/test_pipeline.py
```
*(All 5 unit & integration tests pass with 100% assertions satisfied.)*

### 4. Retrain Models & Benchmark (Optional)
```powershell
python train.py
```
*(Automatically downloads dataset, trains all 7 models with 5-fold GridSearchCV, evaluates metrics, and serializes artifacts in ~16 seconds.)*

### 5. Launch the Web Dashboard
```powershell
streamlit run app.py
```
The interactive application will launch in your browser at `http://localhost:8501`.

---

## 📖 Feature Dictionary & Clinical Normal Ranges

| Feature Code | Clinical Description | Unit / Scale | Normal / Desirable Reference | Diagnostic Relevance |
|---|---|---|---|---|
| `age` | Patient Age | Years | $<45$ (M), $<55$ (F) | Non-modifiable vascular aging index. |
| `sex` | Biological Sex | Binary (0=F, 1=M) | Female (estrogen cardioprotection) | Earlier atherosclerotic onset in males. |
| `cp` | Chest Pain Type | 1-4 | 1: Typical, 2: Atypical, 3: Non-Anginal, 4: Asymptomatic | Silent ischemia (`cp=4`) frequently masks multi-vessel CAD. |
| `trestbps` | Resting Systolic BP | mm Hg | $<120$ mm Hg (AHA Normal) | Arterial wall shear stress and afterload. |
| `chol` | Total Serum Cholesterol | mg/dL | $<200$ mg/dL | Circulating atherogenic lipoprotein particles. |
| `fbs` | Fasting Blood Sugar $>120$ | Binary (0/1) | $\le 120$ mg/dL | Marker of diabetes and metabolic syndrome. |
| `restecg` | Resting 12-Lead ECG | 0-2 | 0: Normal Sinus Rhythm | Repolarization abnormalities / Left Ventricular Hypertrophy. |
| `thalach` | Peak Heart Rate Achieved | bpm | $\ge 85\\%$ Age-predicted max | Chronotropic incompetence signifies coronary insufficiency. |
| `exang` | Exercise-Induced Angina | Binary (0/1) | 0: No exertional angina | Hallmark indicator of myocardial oxygen supply/demand mismatch. |
| `oldpeak` | Exercise ST Depression | mm | $<1.0$ mm | Subendocardial ischemia marker during Bruce protocol. |
| `slope` | Slope of Peak Exercise ST | 1-3 | 1: Upsloping | Flat (2) or Downsloping (3) reflects significant CAD. |
| `ca` | Major Fluoroscopy Vessels | 0-3 | 0 opacified vessels | Direct radiographic calcification and luminal stenosis. |
| `thal` | Nuclear Thallium Scan | 3, 6, 7 | 3: Normal Perfusion | 6: Fixed Scar (Prior MI), 7: Reversible Ischemia (Salvageable). |

---

## 📜 Citations & Academic References
1. **Detrano, R., et al. (1989).** *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64(5), 304–310.
2. **Lundberg, S. M., & Lee, S. I. (2017).** *A unified approach to interpreting model predictions.* Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 4765–4774.
3. **American Heart Association / American College of Cardiology (AHA/ACC).** *2019 Guideline on the Primary Prevention of Cardiovascular Disease.* Circulation, 140(11), e596–e646.

---

  
*HeartGuard AI – Intelligent Heart Disease Prediction & Risk Analysis System*
