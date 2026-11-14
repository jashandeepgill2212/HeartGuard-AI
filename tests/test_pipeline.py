"""
HeartGuard AI - Automated Unit & Integration Tests
Validates data ingestion, model inference, explainable AI (SHAP),
report generation, and batch processing pipelines.
"""

import sys
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import pytest
import pandas as pd
from src.data_loader import load_raw_dataset, audit_and_clean_dataset
from src.ml_pipeline import predict_single_patient, predict_batch_patients
from src.xai_shap import explain_single_patient_shap, compute_global_shap_artifacts
from src.report_generator import generate_pdf_clinical_report
from src.config import CLINICAL_PRESETS, CLEAN_DATASET_FILE


def test_data_loading_and_cleaning():
    """Validates raw data loading, shape, and target binarization."""
    df_raw = load_raw_dataset()
    assert df_raw.shape[0] == 303
    assert df_raw.shape[1] == 14

    df_clean, summary = audit_and_clean_dataset(df_raw)
    assert set(df_clean["target"].unique()).issubset({0, 1})
    assert summary["duplicate_records"] == 0
    assert summary["rows_with_missing"] == 6


def test_model_inference():
    """Validates single patient inference for both low-risk and high-risk presets."""
    low_risk = CLINICAL_PRESETS["Healthy Low-Risk Individual"]
    high_risk = CLINICAL_PRESETS["Severe Multi-Vessel CAD Geriatric Patient"]

    res_low = predict_single_patient(low_risk, model_name="best")
    assert res_low["prediction"] in [0, 1]
    assert 0.0 <= res_low["probability_disease"] <= 1.0
    assert 0.0 <= res_low["confidence_score"] <= 100.0

    res_high = predict_single_patient(high_risk, model_name="best")
    assert res_high["prediction"] == 1
    assert res_high["probability_disease"] > 0.70
    assert res_high["risk_level"] in ["Elevated Risk", "High / Critical Risk"]


def test_shap_explanation():
    """Validates SHAP attribution, base value, and factor decomposition."""
    patient = CLINICAL_PRESETS["High-Risk Ischemic Patient"]
    explanation = explain_single_patient_shap(patient, top_n=6)

    assert "base_value" in explanation
    assert "top_contributions" in explanation
    assert len(explanation["top_contributions"]) <= 6
    assert explanation["plotly_waterfall"] is not None
    assert "CLINICAL CAUSATION" in explanation["disclaimer"]


def test_pdf_report_generation():
    """Validates ReportLab PDF generation and file signature."""
    patient = CLINICAL_PRESETS["Healthy Low-Risk Individual"]
    pred = predict_single_patient(patient, model_name="best")
    xai = explain_single_patient_shap(patient, top_n=5)

    pdf_bytes = generate_pdf_clinical_report(patient, pred, xai)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF magic byte signature
    assert pdf_bytes.startswith(b"%PDF")


def test_batch_prediction():
    """Validates batch vectorized prediction."""
    df_clean = pd.read_csv(CLEAN_DATASET_FILE).head(10)
    batch_res = predict_batch_patients(df_clean, model_name="best")

    assert "predicted_class" in batch_res.columns
    assert "disease_probability" in batch_res.columns
    assert "risk_category" in batch_res.columns
    assert len(batch_res) == 10


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_pipeline.py"])
