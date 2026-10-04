"""
Setup and Verification Script for Predictive Maintenance Dashboard
Predictive Maintenance ML Capstone Project

This script verifies that the trained model bundle and required processed artifacts
exist for the Streamlit dashboard. If any artifact is absent, it executes the pipeline
to generate them.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(_ROOT, "models", "predictive_maintenance_model.joblib")
PROCESSED_DATA_FILE = os.path.join(_ROOT, "data", "processed_fd001.csv")
RAW_DATA_FILE = os.path.join(_ROOT, "data", "train_FD001.txt")


def check_and_setup():
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE PROJECT ENVIRONMENT VERIFICATION")
    print("=" * 60)

    # 1. Check raw dataset
    if not os.path.exists(RAW_DATA_FILE):
        print(f"[!] Raw dataset missing at {RAW_DATA_FILE}")
        sys.exit(1)
    else:
        print(f"[OK] Raw dataset present: {RAW_DATA_FILE}")

    # 2. Check processed dataset
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"[!] Processed dataset missing. Running preprocessing pipeline...")
        from src.preprocessing import run_pipeline as run_prep
        run_prep()
    else:
        print(f"[OK] Processed dataset present: {PROCESSED_DATA_FILE}")

    # 3. Check trained model bundle
    if not os.path.exists(MODEL_FILE):
        print(f"[!] Model bundle missing. Running model training pipeline...")
        import train
        train.run_pipeline()
    else:
        print(f"[OK] Champion model bundle present: {MODEL_FILE}")

    # 4. Verify inference test
    from src.predict import load_model_bundle
    bundle = load_model_bundle(MODEL_FILE)
    print(f"[OK] Successfully verified model bundle: {bundle['model_name']} (Threshold: {bundle.get('prediction_threshold', 0.40)})")
    print("=" * 60)
    print("All systems verified! You can now launch the dashboard:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == '__main__':
    check_and_setup()
