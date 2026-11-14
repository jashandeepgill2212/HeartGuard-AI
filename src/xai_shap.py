"""
HeartGuard AI - Explainable AI (XAI) Suite Using SHAP
Provides global feature attribution, summary plots, individual patient waterfall breakdowns,
positive/negative risk factor decomposition, and clinical causation disclaimers.
"""

import logging
import sys
import types
from pathlib import Path
from typing import Dict, Any, List, Tuple

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

# --- Resilient SHAP Numba Shim for Windows AppLocker environments ---
if "numba" not in sys.modules:
    try:
        import numba
    except Exception:
        # Construct lightweight mock if system blocks _box.pyd DLL
        mock_numba = types.ModuleType("numba")
        mock_numba.njit = lambda *args, **kwargs: (lambda f: f)
        mock_numba.jit = lambda *args, **kwargs: (lambda f: f)
        mock_numba.types = types.ModuleType("numba.types")
        mock_numba.typed = types.ModuleType("numba.typed")
        mock_numba.typed.List = list
        mock_numba.typed.Dict = dict
        mock_numba.experimental = types.ModuleType("numba.experimental")
        mock_numba.experimental.jitclass = lambda *args, **kwargs: (lambda cls: cls)

        sys.modules["numba"] = mock_numba
        sys.modules["numba.types"] = mock_numba.types
        sys.modules["numba.typed"] = mock_numba.typed
        sys.modules["numba.experimental"] = mock_numba.experimental
        sys.modules["numba.experimental.jitclass"] = mock_numba.experimental.jitclass

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from src.config import (
    ARTIFACTS_DIR,
    FEATURE_ORDER,
    FEATURE_METADATA,
    THEME_COLORS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CAUSATION_DISCLAIMER = (
    "CLINICAL CAUSATION VS. STATISTICAL EXPLANATION DISCLAIMER: "
    "SHAP (SHapley Additive exPlanations) values quantify the directional mathematical contribution "
    "of each physiological parameter toward this specific machine learning algorithm's output. "
    "A high positive or negative SHAP value reflects predictive correlation within the training cohort, "
    "NOT necessarily direct biological or pathological etiology. Machine learning explanations must never "
    "replace board-certified clinical judgment, coronary angiography, or clinical laboratory confirmation."
)


def format_feature_label(raw_name: str) -> str:
    """Converts internal feature column names to clinical descriptions."""
    clean_key = raw_name.replace("num__", "").replace("cat__", "")
    clinical_map = {
        "age": "Patient Age",
        "trestbps": "Resting Blood Pressure",
        "chol": "Serum Cholesterol",
        "thalach": "Max Heart Rate Achieved",
        "oldpeak": "ST Depression (Oldpeak)",
        "sex_0.0": "Biological Sex: Female",
        "sex_1.0": "Biological Sex: Male",
        "cp_1.0": "Chest Pain: Typical Angina",
        "cp_2.0": "Chest Pain: Atypical Angina",
        "cp_3.0": "Chest Pain: Non-Anginal Pain",
        "cp_4.0": "Chest Pain: Asymptomatic Ischemia",
        "fbs_0.0": "Fasting Blood Sugar <= 120 mg/dL",
        "fbs_1.0": "Fasting Blood Sugar > 120 mg/dL",
        "restecg_0.0": "Resting ECG: Normal",
        "restecg_1.0": "Resting ECG: ST-T Wave Abnormality",
        "restecg_2.0": "Resting ECG: Left Ventricular Hypertrophy",
        "exang_0.0": "Exercise Angina: None",
        "exang_1.0": "Exercise Angina: Present",
        "slope_1.0": "ST Slope: Upsloping (Physiologic)",
        "slope_2.0": "ST Slope: Flat (Ischemia Risk)",
        "slope_3.0": "ST Slope: Downsloping (Severe Ischemia)",
        "ca_0.0": "Major Vessels Colored: 0 (Normal)",
        "ca_1.0": "Major Vessels Colored: 1 Stenosis",
        "ca_2.0": "Major Vessels Colored: 2 Stenoses",
        "ca_3.0": "Major Vessels Colored: 3 Stenoses",
        "thal_3.0": "Thallium: Normal Perfusion",
        "thal_6.0": "Thallium: Fixed Defect (Scar)",
        "thal_7.0": "Thallium: Reversible Defect (Ischemia)",
    }
    if clean_key in clinical_map:
        return clinical_map[clean_key]
    return clean_key.replace("_", " ").title()


def load_best_model_and_data():
    """Loads best model, preprocessing pipeline, and training/testing sets."""
    model_path = ARTIFACTS_DIR / "best_model.joblib"
    X_train_path = ARTIFACTS_DIR / "X_train.joblib"
    X_test_path = ARTIFACTS_DIR / "X_test.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Model artifacts not found. Please run ml_pipeline.py first.")

    pipeline = joblib.load(model_path)
    X_train = joblib.load(X_train_path)
    X_test = joblib.load(X_test_path)
    return pipeline, X_train, X_test


def compute_global_shap_artifacts() -> Dict[str, Any]:
    """
    Computes and saves global SHAP values across the holdout test set using the best model.
    """
    logger.info("Computing Global SHAP Explanations...")
    pipeline, X_train, X_test = load_best_model_and_data()

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform data through preprocessing pipeline
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = joblib.load(ARTIFACTS_DIR / "transformed_feature_names.joblib")

    # Use TreeExplainer for Tree-based models (Random Forest, XGBoost, Gradient Boosting)
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test_trans)
    except Exception as e:
        logger.warning(f"TreeExplainer fallback to KernelExplainer due to: {e}")
        # Sample background for speed
        background = shap.sample(X_train_trans, 50)
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        shap_values = explainer.shap_values(X_test_trans)

    # For binary classification with TreeExplainer, shap_values can be a list of 2 arrays or 3D array
    if isinstance(shap_values, list):
        # Disease class (index 1)
        shap_matrix = np.array(shap_values[1])
        base_val = float(explainer.expected_value[1]) if hasattr(explainer.expected_value, "__iter__") else float(explainer.expected_value)
    elif len(shap_values.shape) == 3:
        shap_matrix = shap_values[:, :, 1]
        base_val = float(explainer.expected_value[1])
    else:
        shap_matrix = shap_values
        base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0

    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

    # Sort descending
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = [float(mean_abs_shap[i]) for i in sorted_idx]

    # Map back to readable names
    clean_feature_labels = [format_feature_label(f) for f in sorted_features]

    # Package global summary data
    global_xai = {
        "feature_names_raw": sorted_features,
        "feature_names_clean": clean_feature_labels,
        "mean_abs_shap": sorted_importance,
        "base_value": base_val,
        "disclaimer": CAUSATION_DISCLAIMER
    }

    joblib.dump(global_xai, ARTIFACTS_DIR / "shap_global_summary.joblib")
    logger.info("Global SHAP values successfully computed and serialized!")
    return global_xai


# Global in-memory cache for explainer
_CACHED_EXPLAINER = None

def get_or_create_explainer(classifier):
    """Retrieves or instantiates TreeExplainer in memory."""
    global _CACHED_EXPLAINER
    if _CACHED_EXPLAINER is None:
        try:
            _CACHED_EXPLAINER = shap.TreeExplainer(classifier)
        except Exception:
            _CACHED_EXPLAINER = shap.Explainer(classifier)
    return _CACHED_EXPLAINER


def explain_single_patient_shap(
    patient_data: Dict[str, Any],
    top_n: int = 8
) -> Dict[str, Any]:
    """
    Computes individual local SHAP attribution for a single patient.
    Decomposes features into risk-increasing vs risk-decreasing contributions.
    Returns structured data and an interactive Plotly waterfall figure.
    """
    pipeline, X_train, _ = load_best_model_and_data()
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = joblib.load(ARTIFACTS_DIR / "transformed_feature_names.joblib")
    explainer = get_or_create_explainer(classifier)

    # Transform patient input
    df_patient = pd.DataFrame([patient_data])[FEATURE_ORDER]
    patient_trans = preprocessor.transform(df_patient)

    # Predict proba
    proba = pipeline.predict_proba(df_patient)[0]
    prob_disease = float(proba[1])

    # Compute SHAP values for this instance
    shap_vals = explainer.shap_values(patient_trans)

    if isinstance(shap_vals, list):
        patient_shap = np.array(shap_vals[1])[0]
        base_val = float(explainer.expected_value[1]) if hasattr(explainer.expected_value, "__iter__") else float(explainer.expected_value)
    elif len(shap_vals.shape) == 3:
        patient_shap = shap_vals[0, :, 1]
        base_val = float(explainer.expected_value[1])
    else:
        patient_shap = shap_vals[0]
        base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0

    # Match with readable feature descriptions
    contributions = []
    for raw_f, sv, val_trans in zip(feature_names, patient_shap, patient_trans[0]):
        display_name = format_feature_label(raw_f)
        if raw_f.startswith("num__"):
            orig_col = raw_f.replace("num__", "")
            unit = FEATURE_METADATA.get(orig_col, {}).get("unit", "")
            patient_val_display = f"{patient_data.get(orig_col, '')} {unit}".strip()
        elif raw_f.startswith("cat__"):
            patient_val_display = "Active (Present)" if val_trans == 1.0 else "Inactive (Absent)"
        else:
            patient_val_display = str(val_trans)

        contributions.append({
            "raw_feature": raw_f,
            "display_name": display_name,
            "patient_value": patient_val_display,
            "shap_value": float(sv),
            "abs_shap": abs(float(sv)),
            "impact": "Increases Risk" if sv > 0 else ("Decreases Risk" if sv < 0 else "Neutral"),
            "color": THEME_COLORS["accent_coral"] if sv > 0 else THEME_COLORS["accent_teal"]
        })

    # Sort by absolute SHAP impact
    contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

    top_contributions = contributions[:top_n]
    positive_factors = [c for c in contributions if c["shap_value"] > 0]
    negative_factors = [c for c in contributions if c["shap_value"] < 0]

    # Create Plotly Interactive Waterfall Figure
    waterfall_names = [c["display_name"] for c in top_contributions][::-1]
    waterfall_vals = [c["shap_value"] for c in top_contributions][::-1]
    bar_colors = [THEME_COLORS["accent_red"] if v > 0 else THEME_COLORS["accent_teal"] for v in waterfall_vals]

    fig = go.Figure(go.Bar(
        x=waterfall_vals,
        y=waterfall_names,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="#FFFFFF", width=1.5)),
        text=[f"{v:+.3f}" for v in waterfall_vals],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Patient SHAP Attribution Breakdown (Risk Contributors)</b>",
            font=dict(size=16, color=THEME_COLORS["primary_dark"])
        ),
        xaxis=dict(
            title="<b>SHAP Impact on Disease Prediction (Log-Odds / Margin Shift)</b>",
            zeroline=True,
            zerolinecolor="#94A3B8",
            zerolinewidth=2,
            gridcolor="#E2E8F0"
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=40, t=50, b=40),
        height=380
    )

    return {
        "base_value": round(base_val, 4),
        "predicted_probability": round(prob_disease, 4),
        "top_contributions": top_contributions,
        "positive_risk_drivers": positive_factors[:5],
        "protective_factors": negative_factors[:5],
        "plotly_waterfall": fig,
        "disclaimer": CAUSATION_DISCLAIMER
    }


def create_global_importance_figure(top_n: int = 12) -> go.Figure:
    """
    Renders an interactive Plotly horizontal bar chart of global mean absolute SHAP values.
    """
    summary_path = ARTIFACTS_DIR / "shap_global_summary.joblib"
    if not summary_path.exists():
        compute_global_shap_artifacts()
    global_xai = joblib.load(summary_path)

    features = global_xai["feature_names_clean"][:top_n][::-1]
    scores = global_xai["mean_abs_shap"][:top_n][::-1]

    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale=[[0, "#0EA5E9"], [0.5, "#0D9488"], [1, "#E63946"]],
            line=dict(color="#FFFFFF", width=1)
        ),
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Global Feature Importance (Mean |SHAP Value| Across Cohort)</b>",
            font=dict(size=16, color=THEME_COLORS["primary_dark"])
        ),
        xaxis=dict(title="<b>Mean Absolute SHAP Value (Impact on Model Output)</b>", gridcolor="#E2E8F0"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=50, t=50, b=40),
        height=450
    )
    return fig


if __name__ == "__main__":
    gx = compute_global_shap_artifacts()
    print("Top 5 Global Features by SHAP:")
    for f, s in zip(gx["feature_names_clean"][:5], gx["mean_abs_shap"][:5]):
        print(f" - {f:35s}: {s:.4f}")

    sample_patient = {
        "age": 62, "sex": 1, "cp": 4, "trestbps": 158, "chol": 284,
        "fbs": 1, "restecg": 1, "thalach": 118, "exang": 1, "oldpeak": 2.6,
        "slope": 2, "ca": 2, "thal": 7
    }
    local_exp = explain_single_patient_shap(sample_patient)
    print("\nLocal Explanation for Sample High-Risk Patient:")
    print(f"Base Value: {local_exp['base_value']}, Disease Proba: {local_exp['predicted_probability']}")
    print("Top Risk Drivers:")
    for d in local_exp["positive_risk_drivers"]:
        print(f" - {d['display_name']} ({d['patient_value']}): +{d['shap_value']:.4f}")
