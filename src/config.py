"""
HeartGuard AI - System Configuration and Clinical Metadata
Contains feature definitions, categorical mappings, model hyperparameter grids,
color palettes, clinical thresholds, and path configurations.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR, STATIC_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# UCI Heart Disease URLs & Files
CLEVELAND_RAW_FILE = RAW_DATA_DIR / "processed.cleveland.data"
CLEAN_DATASET_FILE = PROCESSED_DATA_DIR / "heart_cleveland_clean.csv"
DATASET_METADATA_FILE = PROCESSED_DATA_DIR / "dataset_summary.json"
SAMPLE_BATCH_FILE = PROCESSED_DATA_DIR / "sample_patient_batch.csv"

UCI_CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)

# Raw Column Names in UCI Cleveland File
RAW_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# Continuous vs Categorical Feature Breakdown
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_ORDER = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

# Feature Descriptions & Clinical Meanings
FEATURE_METADATA = {
    "age": {
        "name": "Age",
        "unit": "years",
        "type": "continuous",
        "description": "Patient age in completed years.",
        "clinical_significance": "Cardiovascular risk escalates sharply after age 45 for males and 55 for females.",
        "min": 25,
        "max": 85,
        "default": 54
    },
    "sex": {
        "name": "Biological Sex",
        "unit": "binary",
        "type": "categorical",
        "description": "Biological sex of the patient.",
        "clinical_significance": "Pre-menopausal females have estrogen cardioprotection; males typically exhibit earlier CAD onset.",
        "options": {0: "Female", 1: "Male"},
        "default": 1
    },
    "cp": {
        "name": "Chest Pain Type",
        "unit": "scale (1-4)",
        "type": "categorical",
        "description": "Type of chest discomfort experienced by the patient.",
        "clinical_significance": "Asymptomatic ischemia or atypical presentations frequently mask severe multi-vessel obstruction.",
        "options": {
            1: "Typical Angina (provoked by stress/exertion, relieved by rest/nitro)",
            2: "Atypical Angina (meets 2 of 3 classical criteria)",
            3: "Non-Anginal Pain (sharp, pleuritic or musculoskeletal)",
            4: "Asymptomatic (silent myocardial ischemia or diabetic neuropathy)"
        },
        "default": 1
    },
    "trestbps": {
        "name": "Resting Blood Pressure",
        "unit": "mm Hg",
        "type": "continuous",
        "description": "Resting arterial systolic blood pressure upon clinical intake.",
        "clinical_significance": "AHA/ACC defines Stage 1 HTN at 130-139 mmHg and Stage 2 at >=140 mmHg, driving arterial remodeling.",
        "min": 80,
        "max": 220,
        "default": 130
    },
    "chol": {
        "name": "Serum Cholesterol",
        "unit": "mg/dL",
        "type": "continuous",
        "description": "Total serum cholesterol level.",
        "clinical_significance": "Values above 200 mg/dL indicate elevated atherogenic lipoprotein deposition and plaque risk.",
        "min": 100,
        "max": 600,
        "default": 240
    },
    "fbs": {
        "name": "Fasting Blood Sugar > 120 mg/dL",
        "unit": "binary",
        "type": "categorical",
        "description": "Fasting venous glucose concentration exceeding 120 mg/dL.",
        "clinical_significance": "Markers of impaired fasting glucose, insulin resistance, and elevated metabolic syndrome index.",
        "options": {0: "No (<= 120 mg/dL)", 1: "Yes (> 120 mg/dL - Prediabetes/Diabetes)"},
        "default": 0
    },
    "restecg": {
        "name": "Resting Electrocardiogram (ECG)",
        "unit": "category (0-2)",
        "type": "categorical",
        "description": "12-lead resting electrocardiogram findings.",
        "clinical_significance": "Identifies baseline electrical abnormalities, repolarization shifts, or ventricular hypertrophy.",
        "options": {
            0: "Normal Sinus Rhythm",
            1: "ST-T Wave Abnormality (T-wave inversion / ST shift > 0.05 mV)",
            2: "Left Ventricular Hypertrophy (probable/definite by Estes' criteria)"
        },
        "default": 0
    },
    "thalach": {
        "name": "Maximum Heart Rate Achieved",
        "unit": "bpm",
        "type": "continuous",
        "description": "Peak heart rate during maximal Bruce treadmill stress protocol.",
        "clinical_significance": "Chronotropic incompetence (inability to reach >=85% predicted max HR) strongly correlates with ischemia.",
        "min": 60,
        "max": 220,
        "default": 150
    },
    "exang": {
        "name": "Exercise-Induced Angina",
        "unit": "binary",
        "type": "categorical",
        "description": "Presence of ischemic chest pain during treadmill stress test.",
        "clinical_significance": "Hallmark clinical indicator of coronary hypoperfusion under elevated myocardial oxygen demand.",
        "options": {0: "No Angina on Exertion", 1: "Yes (Exertional Angina Triggered)"},
        "default": 0
    },
    "oldpeak": {
        "name": "ST Depression (Oldpeak)",
        "unit": "mm",
        "type": "continuous",
        "description": "ST-segment depression induced by exercise relative to resting baseline.",
        "clinical_significance": "ST depressions > 1.0 mm signify subendocardial ischemia; values > 2.0 mm denote severe coronary insufficiency.",
        "min": 0.0,
        "max": 6.5,
        "default": 1.0
    },
    "slope": {
        "name": "Slope of Peak Exercise ST Segment",
        "unit": "category (1-3)",
        "type": "categorical",
        "description": "Morphology of ST segment at maximal workload.",
        "clinical_significance": "Horizontal (flat) or downsloping ST segments have high diagnostic specificity for coronary artery disease.",
        "options": {
            1: "Upsloping (frequently physiologic/benign)",
            2: "Flat (horizontal depression - typical ischemic pattern)",
            3: "Downsloping (severe multi-vessel or left main coronary lesion)"
        },
        "default": 1
    },
    "ca": {
        "name": "Major Vessels Colored by Fluoroscopy",
        "unit": "count (0-3)",
        "type": "categorical",
        "description": "Number of major epicardial coronary vessels visualized via fluoroscopy.",
        "clinical_significance": "Higher calcification or vessel stenosis counts directly mirror coronary atherosclerotic burden.",
        "options": {
            0: "0 Vessels Colored",
            1: "1 Vessel Colored",
            2: "2 Vessels Colored",
            3: "3 Vessels Colored"
        },
        "default": 0
    },
    "thal": {
        "name": "Thallium-201 Scintigraphy",
        "unit": "category",
        "type": "categorical",
        "description": "Nuclear thallium stress perfusion scan defect status.",
        "clinical_significance": "Reversible defects signify salvageable ischemic myocardium; fixed defects represent prior myocardial infarction scar.",
        "options": {
            3: "Normal Myocardial Perfusion",
            6: "Fixed Defect (Non-viable necrotic scar / prior infarct)",
            7: "Reversible Defect (Exercise-induced ischemia / viable tissue at risk)"
        },
        "default": 3
    }
}

# Clinical Patient Presets for Quick Testing
CLINICAL_PRESETS = {
    "Healthy Low-Risk Individual": {
        "age": 36,
        "sex": 0,
        "cp": 3,
        "trestbps": 114,
        "chol": 178,
        "fbs": 0,
        "restecg": 0,
        "thalach": 182,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
        "profile": "36-year-old female athlete with normal blood pressure, optimal lipids, and excellent exercise tolerance."
    },
    "Moderate-Risk Hypertensive Patient": {
        "age": 52,
        "sex": 1,
        "cp": 2,
        "trestbps": 138,
        "chol": 235,
        "fbs": 0,
        "restecg": 0,
        "thalach": 146,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 1,
        "thal": 3,
        "profile": "52-year-old male with Stage 1 hypertension, borderline hypercholesterolemia, and moderate ST depression."
    },
    "High-Risk Ischemic Patient": {
        "age": 62,
        "sex": 1,
        "cp": 4,
        "trestbps": 158,
        "chol": 284,
        "fbs": 1,
        "restecg": 1,
        "thalach": 118,
        "exang": 1,
        "oldpeak": 2.6,
        "slope": 2,
        "ca": 2,
        "thal": 7,
        "profile": "62-year-old male diabetic smoker with exertional angina, significant ST depression, and reversible perfusion defect."
    },
    "Severe Multi-Vessel CAD Geriatric Patient": {
        "age": 68,
        "sex": 1,
        "cp": 4,
        "trestbps": 168,
        "chol": 298,
        "fbs": 1,
        "restecg": 2,
        "thalach": 104,
        "exang": 1,
        "oldpeak": 3.4,
        "slope": 3,
        "ca": 3,
        "thal": 7,
        "profile": "68-year-old male with left ventricular hypertrophy, 3 fluoroscopy-opacified stenotic vessels, and downsloping ST."
    }
}

# Model List
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "Decision Tree",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "Gradient Boosting",
    "XGBoost"
]

# Primary UI Theme Colors
THEME_COLORS = {
    "primary_dark": "#0B192C",
    "secondary_dark": "#1E3E62",
    "card_bg": "#FFFFFF",
    "page_bg": "#F8FAFC",
    "accent_teal": "#00ADB5",
    "accent_cyan": "#0EA5E9",
    "accent_coral": "#FF6B6B",
    "accent_red": "#E63946",
    "accent_green": "#10B981",
    "accent_amber": "#F59E0B",
    "text_dark": "#1E293B",
    "text_muted": "#64748B",
    "border_color": "#E2E8F0"
}
