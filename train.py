"""
HeartGuard AI - Standalone Training & Benchmark Orchestrator
Executes end-to-end data ingestion, model tuning, cross-validation,
metrics serialization, and explainability artifact computation.
"""

import sys
import time
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

from src.data_loader import prepare_dataset_pipeline
from src.ml_pipeline import train_and_evaluate_all_models
from src.xai_shap import compute_global_shap_artifacts


def main():
    print("=" * 80)
    print("HEARTGUARD AI: CLINICAL ML PIPELINE TRAINING & BENCHMARKING")
    print("=" * 80)

    start_time = time.time()

    # Step 1: Data Pipeline
    print("\n[1/3] Ingesting and Auditing UCI Cleveland Heart Disease Dataset...")
    df, summary = prepare_dataset_pipeline()
    print(f" -> Processed {len(df)} records across {summary['total_features']} features.")
    print(f" -> Target balance: {summary['target_balance_ratio']['no_disease_pct']}% Healthy vs {summary['target_balance_ratio']['disease_pct']}% CAD.")
    print(f" -> Imputation strategy: Median for continuous, Most Frequent for categorical.")

    # Step 2: Model Training & Tuning
    print("\n[2/3] Training & Tuning 7 Machine Learning Algorithms (Stratified 5-Fold CV)...")
    ml_output = train_and_evaluate_all_models(test_size=0.20, random_state=42)
    results = ml_output["results"]
    best_model = ml_output["best_model_name"]

    print("\n" + "-" * 95)
    print(f"{'Model':26s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':8s} | {'Specificity':11s} | {'ROC-AUC':8s} | {'F1':8s}")
    print("-" * 95)
    for model_name, m in results.items():
        print(f"{model_name:26s} | {m['accuracy']:<8.4f} | {m['precision']:<9.4f} | {m['recall']:<8.4f} | {m['specificity']:<11.4f} | {m['roc_auc']:<8.4f} | {m['f1']:<8.4f}")
    print("-" * 95)
    print(f"\n[*] Best Model Selected: {best_model} (Composite Score: {results[best_model]['clinical_composite_score']})")

    # Step 3: Explainable AI Suite
    print("\n[3/3] Computing Global Explainable AI (SHAP) Attributions...")
    global_xai = compute_global_shap_artifacts()
    print(f" -> Successfully serialized SHAP global explanations for {len(global_xai['feature_names_raw'])} encoded features.")

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"PIPELINE EXECUTION COMPLETE in {elapsed}s! All artifacts serialized to models/artifacts/")
    print("=" * 80)


if __name__ == "__main__":
    main()
