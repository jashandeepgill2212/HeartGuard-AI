"""
HeartGuard AI - Data Ingestion, Cleaning, and Validation Module
Handles automated downloading, schema validation, missing value detection,
duplicate auditing, target binarization, and exploratory summary generation.
"""

import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Tuple, Any

# Ensure project root is in sys.path
BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import numpy as np
import pandas as pd

from src.config import (
    CLEVELAND_RAW_FILE,
    CLEAN_DATASET_FILE,
    DATASET_METADATA_FILE,
    SAMPLE_BATCH_FILE,
    UCI_CLEVELAND_URL,
    RAW_COLUMNS,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
    RAW_DATA_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_uci_dataset(force: bool = False) -> Path:
    """
    Downloads the official Cleveland Heart Disease dataset from UCI Machine Learning Repository
    if not already present locally.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CLEVELAND_RAW_FILE.exists() and not force:
        logger.info(f"Dataset already present at {CLEVELAND_RAW_FILE}")
        return CLEVELAND_RAW_FILE

    logger.info(f"Downloading UCI Heart Disease dataset from {UCI_CLEVELAND_URL}...")
    try:
        urllib.request.urlretrieve(UCI_CLEVELAND_URL, CLEVELAND_RAW_FILE)
        logger.info(f"Successfully downloaded to {CLEVELAND_RAW_FILE} ({CLEVELAND_RAW_FILE.stat().st_size} bytes)")
    except Exception as e:
        logger.warning(f"Network download failed: {e}. Checking local cache...")
        if not CLEVELAND_RAW_FILE.exists():
            raise FileNotFoundError(
                f"Could not download or find dataset at {CLEVELAND_RAW_FILE}. Ensure network access or provide file manually."
            )
    return CLEVELAND_RAW_FILE


def load_raw_dataset() -> pd.DataFrame:
    """
    Reads the raw UCI Cleveland file, replacing '?' with NaN and applying standard column headers.
    """
    download_uci_dataset()
    df = pd.read_csv(
        CLEVELAND_RAW_FILE,
        names=RAW_COLUMNS,
        na_values=["?", "NA", "null", ""],
        sep=",",
        skipinitialspace=True
    )
    logger.info(f"Loaded raw dataset with shape: {df.shape}")
    return df


def audit_and_clean_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs data quality audit:
    - Analyzes missing value distribution.
    - Audits duplicate records.
    - Converts multi-class angiographic target (0-4) to binary CAD status (0 vs 1).
    - Checks physiological plausibility of clinical features.
    - Generates metadata report for dashboard display.
    """
    df = df_raw.copy()

    # 1. Duplicate detection
    num_duplicates = int(df.duplicated().sum())

    # 2. Missing value detection
    missing_dict = df.isna().sum().to_dict()
    total_missing_cells = int(df.isna().sum().sum())
    rows_with_missing = int(df.isna().any(axis=1).sum())

    # 3. Target analysis & binarization
    original_target_dist = df["target"].value_counts().sort_index().to_dict()
    # In clinical literature (Detrano et al.):
    # 0 = absence of significant CAD (<50% stenosis)
    # 1, 2, 3, 4 = presence of significant CAD (>=50% stenosis in 1, 2, 3, or 4 vessels)
    df["target_original"] = df["target"].astype(int)
    df["target"] = (df["target_original"] > 0).astype(int)
    binary_target_dist = df["target"].value_counts().to_dict()

    # 4. Range and validity checks
    validity_checks = {
        "age_valid": bool(((df["age"] >= 18) & (df["age"] <= 120)).all()),
        "trestbps_valid": bool(((df["trestbps"] >= 60) & (df["trestbps"] <= 260)).all()),
        "chol_valid": bool(((df["chol"] >= 50) & (df["chol"] <= 700)).all()),
        "thalach_valid": bool(((df["thalach"] >= 50) & (df["thalach"] <= 250)).all()),
        "oldpeak_valid": bool(((df["oldpeak"] >= 0.0) & (df["oldpeak"] <= 10.0)).all()),
    }

    # 5. Summary metrics
    summary = {
        "dataset_name": "UCI Cleveland Heart Disease Database",
        "citation": "Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64(5), 304-310.",
        "total_records": int(len(df)),
        "total_features": int(len(RAW_COLUMNS) - 1),
        "continuous_features": CONTINUOUS_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "duplicate_records": num_duplicates,
        "total_missing_cells": total_missing_cells,
        "rows_with_missing": rows_with_missing,
        "missing_per_feature": missing_dict,
        "target_conversion_explanation": (
            "The clinical gold-standard angiographic assessment records 0 for coronary narrowing <50%, "
            "and 1-4 for narrowing >=50% across 1, 2, 3, or 4 major coronary arteries. "
            "For binary risk classification, 0 designates Healthy / Low Risk, and 1-4 are mapped to 1 (CAD Present)."
        ),
        "original_target_distribution": original_target_dist,
        "binary_target_distribution": binary_target_dist,
        "target_balance_ratio": {
            "no_disease_pct": round((binary_target_dist.get(0, 0) / len(df)) * 100, 2),
            "disease_pct": round((binary_target_dist.get(1, 0) / len(df)) * 100, 2),
        },
        "validity_audit": validity_checks,
        "feature_stats": {}
    }

    for col in CONTINUOUS_FEATURES:
        summary["feature_stats"][col] = {
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2),
            "min": round(float(df[col].min()), 2),
            "median": round(float(df[col].median()), 2),
            "max": round(float(df[col].max()), 2),
            "iqr": round(float(df[col].quantile(0.75) - df[col].quantile(0.25)), 2)
        }

    return df, summary


def generate_sample_batch_file(df: pd.DataFrame) -> Path:
    """
    Creates a sample batch CSV file representing 15 diverse patients for the Batch Prediction module.
    """
    sample_indices = [
        0, 1, 2, 3, 4, 10, 15, 20, 25, 30, 45, 55, 65, 80, 100
    ]
    sample_df = df.iloc[sample_indices].copy()
    sample_df["patient_id"] = [f"PT-{1000 + i}" for i in range(len(sample_df))]

    # Reorder columns: patient_id first, then clinical features, keep ground truth for benchmarking
    cols = ["patient_id"] + [col for col in RAW_COLUMNS if col in sample_df.columns]
    sample_df = sample_df[cols]

    sample_df.to_csv(SAMPLE_BATCH_FILE, index=False)
    logger.info(f"Generated sample patient batch at {SAMPLE_BATCH_FILE}")
    return SAMPLE_BATCH_FILE


def prepare_dataset_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full data pipeline execution: loads raw data, audits quality, saves cleaned CSV,
    generates metadata JSON, and creates sample batch data.
    """
    df_raw = load_raw_dataset()
    df_clean, summary = audit_and_clean_dataset(df_raw)

    # Save clean dataset
    df_clean.to_csv(CLEAN_DATASET_FILE, index=False)
    logger.info(f"Cleaned dataset saved to {CLEAN_DATASET_FILE}")

    # Save summary metadata
    with open(DATASET_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Dataset metadata saved to {DATASET_METADATA_FILE}")

    # Create batch sample
    generate_sample_batch_file(df_clean)

    return df_clean, summary


if __name__ == "__main__":
    df, summary = prepare_dataset_pipeline()
    print("Dataset successfully processed!")
    print(f"Total rows: {len(df)}, Features: {summary['total_features']}")
    print(f"Target distribution: {summary['binary_target_distribution']}")
    print(f"Missing count: {summary['missing_per_feature']}")
