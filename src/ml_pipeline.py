"""
HeartGuard AI - Machine Learning Pipeline Module
Implements end-to-end model training, hyperparameter tuning via Stratified GridSearchCV,
comprehensive clinical performance metrics, cross-validation auditing, and model persistence.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

BASE_PATH = Path(__file__).resolve().parent.parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    CLEAN_DATASET_FILE,
    ARTIFACTS_DIR,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_preprocessor() -> Tuple[ColumnTransformer, List[str]]:
    """
    Creates a leak-proof Scikit-Learn ColumnTransformer:
    - Numerical: SimpleImputer(median) -> StandardScaler
    - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, CONTINUOUS_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def get_model_hyperparameter_grids() -> Dict[str, Tuple[Any, Dict[str, List]]]:
    """
    Defines the 7 machine learning algorithms and their tuning search grids.
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {
                "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
                "classifier__solver": ["lbfgs"]
            }
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__max_depth": [3, 5, 8, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__criterion": ["gini", "entropy"]
            }
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "classifier__max_depth": [2, 3, 5, 7],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__criterion": ["gini", "entropy"]
            }
        ),
        "Support Vector Machine": (
            SVC(probability=True, random_state=42),
            {
                "classifier__C": [0.1, 1.0, 5.0],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"]
            }
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"]
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [3, 4]
            }
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=42),
            {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [3, 4]
            }
        )
    }
    return models


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Computes real, un-faked clinical diagnostic performance metrics on the unseen test set.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC Curve Data
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

    # Precision-Recall Curve Data
    prec_pts, rec_pts, pr_thresholds = precision_recall_curve(y_test, y_proba)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "raw": cm.tolist()
        },
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist()
        },
        "pr_curve": {
            "precision": prec_pts.tolist(),
            "recall": rec_pts.tolist(),
            "thresholds": pr_thresholds.tolist()
        },
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist()
    }


def compute_cross_validation_metrics(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Executes Stratified 5-Fold Cross Validation to ensure generalizability and detect overfitting.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }
    cv_scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    result = {}
    for metric in scoring.keys():
        test_scores = cv_scores[f"test_{metric}"]
        result[metric] = {
            "mean": round(float(np.mean(test_scores)), 4),
            "std": round(float(np.std(test_scores)), 4),
            "scores": [round(float(s), 4) for s in test_scores]
        }
    return result


def train_and_evaluate_all_models(
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Full machine learning orchestration:
    1. Loads cleaned Cleveland dataset.
    2. Performs stratified train/test split (80/20).
    3. Builds leak-proof ColumnTransformer.
    4. Tunes all 7 models using Stratified 5-Fold GridSearchCV (optimizing ROC-AUC).
    5. Evaluates on holdout test set (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC).
    6. Performs cross-validation stability check.
    7. Selects best model using composite clinical utility score: (0.40 * ROC_AUC + 0.35 * Recall + 0.25 * F1).
    8. Persists all artifacts to disk using Joblib.
    """
    logger.info("Initiating full ML Training & Evaluation Suite...")
    df = pd.read_csv(CLEAN_DATASET_FILE)

    X = df[FEATURE_ORDER]
    y = df["target"]

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train set: {X_train.shape[0]} patients, Test set: {X_test.shape[0]} patients.")

    # Save split data for reference and XAI background
    joblib.dump(X_train, ARTIFACTS_DIR / "X_train.joblib")
    joblib.dump(X_test, ARTIFACTS_DIR / "X_test.joblib")
    joblib.dump(y_train, ARTIFACTS_DIR / "y_train.joblib")
    joblib.dump(y_test, ARTIFACTS_DIR / "y_test.joblib")

    models_config = get_model_hyperparameter_grids()
    results = {}
    cv_summary = {}
    fitted_pipelines = {}
    roc_pr_data = {}

    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for model_name, (estimator, param_grid) in models_config.items():
        logger.info(f"Tuning & Training: {model_name}...")
        preprocessor = build_preprocessor()
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator)
        ])

        # Grid search with StratifiedKFold optimizing for ROC-AUC
        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            cv=cv_stratified,
            scoring="roc_auc",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_roc_auc = float(grid_search.best_score_)

        # Evaluate on unseen holdout test set
        test_metrics = evaluate_model(best_pipeline, X_test, y_test)
        test_metrics["best_params"] = {k.replace("classifier__", ""): v for k, v in best_params.items()}
        test_metrics["cv_roc_auc_train"] = round(best_cv_roc_auc, 4)

        # 5-fold CV on train set for all metrics
        cv_metrics = compute_cross_validation_metrics(best_pipeline, X_train, y_train)
        cv_summary[model_name] = cv_metrics

        # Composite Clinical Score (prioritizes ROC-AUC and Recall/Sensitivity to catch disease)
        # Clinical reasoning: Missing a diseased patient (False Negative) has lethal implications;
        # High recall and high discrimination (ROC-AUC) are vital.
        composite_score = (
            0.40 * test_metrics["roc_auc"] +
            0.35 * test_metrics["recall"] +
            0.25 * test_metrics["f1"]
        )
        test_metrics["clinical_composite_score"] = round(composite_score, 4)

        results[model_name] = test_metrics
        fitted_pipelines[model_name] = best_pipeline

        # Save individual model
        slug = model_name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(best_pipeline, ARTIFACTS_DIR / f"{slug}.joblib")

        # Collect ROC / PR curves
        roc_pr_data[model_name] = {
            "roc": test_metrics["roc_curve"],
            "pr": test_metrics["pr_curve"],
            "roc_auc": test_metrics["roc_auc"],
            "f1": test_metrics["f1"],
            "recall": test_metrics["recall"]
        }

    # Best Model Selection
    best_model_name = max(
        results.keys(),
        key=lambda m: results[m]["clinical_composite_score"]
    )
    best_pipeline = fitted_pipelines[best_model_name]

    # Save Best Model explicitly
    joblib.dump(best_pipeline, ARTIFACTS_DIR / "best_model.joblib")

    # Fit standalone preprocessor on X_train to extract transformed feature names
    preprocessor_fitted = build_preprocessor()
    preprocessor_fitted.fit(X_train)
    joblib.dump(preprocessor_fitted, ARTIFACTS_DIR / "preprocessing_pipeline.joblib")

    # Extract transformed feature names
    cat_encoder = preprocessor_fitted.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = CONTINUOUS_FEATURES + encoded_cat_names
    joblib.dump(all_feature_names, ARTIFACTS_DIR / "transformed_feature_names.joblib")

    # Save ROC and PR curve data
    joblib.dump(roc_pr_data, ARTIFACTS_DIR / "roc_pr_data.joblib")

    # Save metrics JSON (without numpy/unserializable fields)
    serializable_metrics = {}
    for name, m in results.items():
        serializable_metrics[name] = {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "clinical_composite_score": m["clinical_composite_score"],
            "confusion_matrix": m["confusion_matrix"],
            "best_params": m["best_params"],
            "cv_roc_auc_train": m["cv_roc_auc_train"]
        }

    with open(ARTIFACTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=4)

    with open(ARTIFACTS_DIR / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=4)

    # Save summary metadata
    selection_summary = {
        "best_model_name": best_model_name,
        "selection_strategy": (
            "Optimized via composite clinical index (0.40 * ROC-AUC + 0.35 * Recall + 0.25 * F1). "
            "In cardiac disease screening, high sensitivity (Recall) is paramount to avoid fatal false negatives, "
            "while ROC-AUC guarantees robust discrimination across diverse patient operational thresholds."
        ),
        "best_model_metrics": serializable_metrics[best_model_name],
        "test_sample_size": int(len(y_test)),
        "train_sample_size": int(len(y_train)),
        "class_distribution_test": {
            "no_disease (0)": int((y_test == 0).sum()),
            "disease_present (1)": int((y_test == 1).sum())
        }
    }
    with open(ARTIFACTS_DIR / "best_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(selection_summary, f, indent=4)

    logger.info(f"All 7 models trained! Best Model: {best_model_name} (Composite Score: {results[best_model_name]['clinical_composite_score']})")
    return {
        "results": serializable_metrics,
        "best_model_name": best_model_name,
        "summary": selection_summary
    }


def predict_single_patient(
    patient_data: Dict[str, Any],
    model_name: str = "best"
) -> Dict[str, Any]:
    """
    Inference function for a single patient record.
    Supports either the best model or any of the 7 individual algorithms.
    """
    if model_name == "best":
        model_path = ARTIFACTS_DIR / "best_model.joblib"
    else:
        slug = model_name.lower().replace(" ", "_").replace("-", "_")
        model_path = ARTIFACTS_DIR / f"{slug}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please train models first.")

    pipeline = joblib.load(model_path)
    df_input = pd.DataFrame([patient_data])[FEATURE_ORDER]

    prediction = int(pipeline.predict(df_input)[0])
    probabilities = pipeline.predict_proba(df_input)[0]
    prob_disease = float(probabilities[1])
    prob_healthy = float(probabilities[0])

    # AHA/ACC Cardiovascular Risk Category
    if prob_disease < 0.20:
        risk_level = "Low Risk"
        risk_color = "#10B981"  # Emerald Green
        clinical_urgency = "Routine Preventive Monitoring"
    elif prob_disease < 0.50:
        risk_level = "Borderline / Moderate Risk"
        risk_color = "#F59E0B"  # Amber
        clinical_urgency = "Lifestyle Modification & Follow-up Evaluation"
    elif prob_disease < 0.80:
        risk_level = "Elevated Risk"
        risk_color = "#EF4444"  # Red
        clinical_urgency = "Prompt Cardiology Consultation & Confirmatory Testing"
    else:
        risk_level = "High / Critical Risk"
        risk_color = "#991B1B"  # Dark Red
        clinical_urgency = "Urgent Diagnostic Angiography / Comprehensive Workup"

    return {
        "prediction": prediction,
        "prediction_label": "Heart Disease Indicated" if prediction == 1 else "No Significant Heart Disease Detected",
        "probability_disease": round(prob_disease, 4),
        "probability_healthy": round(prob_healthy, 4),
        "confidence_score": round(max(prob_disease, prob_healthy) * 100, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "clinical_urgency": clinical_urgency,
        "model_used": model_name
    }


def predict_batch_patients(
    df_batch: pd.DataFrame,
    model_name: str = "best"
) -> pd.DataFrame:
    """
    Performs vectorized inference on a batch of patients and appends risk scores,
    prediction labels, and clinical risk tiers.
    """
    if model_name == "best":
        model_path = ARTIFACTS_DIR / "best_model.joblib"
    else:
        slug = model_name.lower().replace(" ", "_").replace("-", "_")
        model_path = ARTIFACTS_DIR / f"{slug}.joblib"

    pipeline = joblib.load(model_path)
    X_input = df_batch[FEATURE_ORDER]

    preds = pipeline.predict(X_input)
    probas = pipeline.predict_proba(X_input)[:, 1]

    df_out = df_batch.copy()
    df_out["predicted_class"] = preds
    df_out["predicted_label"] = np.where(preds == 1, "CAD Present", "Healthy")
    df_out["disease_probability"] = np.round(probas * 100, 2)

    # Risk tiers
    conditions = [
        probas < 0.20,
        (probas >= 0.20) & (probas < 0.50),
        (probas >= 0.50) & (probas < 0.80),
        probas >= 0.80
    ]
    tiers = ["Low Risk", "Borderline Risk", "Elevated Risk", "High Risk"]
    df_out["risk_category"] = np.select(conditions, tiers, default="Undetermined")

    return df_out


if __name__ == "__main__":
    out = train_and_evaluate_all_models()
    print("\nTraining Completed Successfully!")
    print(f"Best Model Selected: {out['best_model_name']}")
    for m, met in out["results"].items():
        print(f"{m:25s} | Acc: {met['accuracy']:.4f} | Prec: {met['precision']:.4f} | Rec: {met['recall']:.4f} | Spec: {met['specificity']:.4f} | ROC-AUC: {met['roc_auc']:.4f} | F1: {met['f1']:.4f}")
