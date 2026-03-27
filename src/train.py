import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from src.database import load_features, save_model_run, get_patient_ids

# features column — same ordering as features.py
FEATURE_COLS = [
    "delta_abs", "theta_abs", "alpha_abs", "sigma_abs", "beta_abs",
    "delta_rel", "theta_rel", "alpha_rel", "sigma_rel", "beta_rel",
    "eeg_variance", "zcr", "delta_beta_ratio",
    "eog_variance", "emg_variance",
]

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

def split_by_patient(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split epochs into train/test sets by patient_id.

    All epochs of a given patient go entirely to train OR test —
    never split across both. This prevents data leakage between
    nights of the same patient.

    Parameters
    ----------
    df         : full features DataFrame from load_features()
    test_ratio : fraction of patients assigned to test set

    Returns
    -------
    df_train, df_test, train_patients, test_patients
    """
    patients = sorted(df["patient_id"].unique())
    n_test   = max(1, int(len(patients) * test_ratio))

    # Reproducible random split
    rng          = np.random.default_rng(seed=42)
    test_patients  = list(rng.choice(patients, size=n_test, replace=False))
    train_patients = [p for p in patients if p not in test_patients]

    df_train = df[df["patient_id"].isin(train_patients)].copy()
    df_test  = df[df["patient_id"].isin(test_patients)].copy()

    return df_train, df_test, train_patients, test_patients

def train(
    algorithm: str = "random_forest",
    test_ratio: float = 0.2,
    n_estimators: int = 200,
    max_depth: int = None,
    notes: str = ""
) -> dict:
    """
    Train a sleep stage classifier and save it to model/.

    Parameters
    ----------
    algorithm    : model name — currently "random_forest"
    test_ratio   : fraction of patients for test set
    n_estimators : number of trees in the forest
    max_depth    : max tree depth (None = unlimited)
    notes        : free text comment saved with the run

    Returns
    -------
    dict with all metrics and paths — same structure as model_runs table
    """
    # --- Load all features from SQLite ---
    df = load_features()
    if df.empty:
        raise ValueError("No features found in database. Run batch extraction first.")

    print(f"Loaded {len(df)} epochs from {df['patient_id'].nunique()} patients.")

    # --- Split by patient ---
    df_train, df_test, train_patients, test_patients = split_by_patient(
        df, test_ratio=test_ratio
    )
    print(f"Train: {len(df_train)} epochs ({len(train_patients)} patients)")
    print(f"Test:  {len(df_test)} epochs ({len(test_patients)} patients)")

    # --- Prepare X and y ---
    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["label"].values
    X_test  = df_test[FEATURE_COLS].values
    y_test  = df_test["label"].values

    # --- Standardise features ---
    # Scaler is fit ONLY on train data — never on test data.
    # This is critical: fitting on test would leak information.
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)   # apply same transform, don't refit

    # --- Train model ---
    # class_weight="balanced" compensates for class imbalance (W >> N1).
    # It weights each class inversely proportional to its frequency.
    if algorithm == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=-1,       # use all CPU cores
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"Training {algorithm} with {n_estimators} trees...")
    model.fit(X_train, y_train)

    # --- Evaluate on test set ---
    y_pred   = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    # F1 per stage — labels 0=W 1=N1 2=N2 3=N3 4=REM
    report = classification_report(
        y_test, y_pred,
        target_names=["W", "N1", "N2", "N3", "REM"],
        output_dict=True,
        zero_division=0
    )
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    print(f"\nAccuracy: {accuracy:.3f}")
    print(classification_report(
        y_test, y_pred,
        target_names=["W", "N1", "N2", "N3", "REM"],
        zero_division=0
    ))

    # --- Save model + scaler to disk ---
    # Both are saved together so the scaler is always applied
    # consistently when loading the model for inference.
    model_name = f"{algorithm}_n{n_estimators}"
    model_path = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    print(f"Model saved to {model_path}")

    # --- Log run to SQLite ---
    run = {
        "algorithm":       algorithm,
        "params":          json.dumps({
            "n_estimators": n_estimators,
            "max_depth":    max_depth,
        }),
        "train_patients":  ",".join(train_patients),
        "test_patients":   ",".join(test_patients),
        "n_train_epochs":  int(len(df_train)),
        "n_test_epochs":   int(len(df_test)),
        "accuracy":        accuracy,
        "f1_weighted":     f1_weighted,
        "f1_W":            float(report["W"]["f1-score"]),
        "f1_N1":           float(report["N1"]["f1-score"]),
        "f1_N2":           float(report["N2"]["f1-score"]),
        "f1_N3":           float(report["N3"]["f1-score"]),
        "f1_REM":          float(report["REM"]["f1-score"]),
        "notes":           notes,
    }
    run_id = save_model_run(run)
    print(f"Run saved to database with id={run_id}")

    return {**run, "run_id": run_id, "model_path": str(model_path)}

def predict(psg_path: str, model_path: str) -> pd.DataFrame:
    """
    Run inference on a single PSG recording using a saved model.

    Parameters
    ----------
    psg_path   : path to the PSG EDF file
    model_path : path to the .pkl file saved by train()

    Returns
    -------
    pd.DataFrame with columns: epoch_index, predicted_label, predicted_name
    """
    from src.features import extract_features

    # Load model and scaler
    bundle = joblib.load(model_path)
    model  = bundle["model"]
    scaler = bundle["scaler"]

    # Extract features from the recording
    # patient_id here is just a placeholder — not saved anywhere
    df = extract_features(psg_path, hypnogram_path=None, patient_id="inference")

    X = df[FEATURE_COLS].values
    X = scaler.transform(X)

    labels      = model.predict(X)
    label_names = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    df["predicted_label"] = labels
    df["predicted_name"]  = [label_names[l] for l in labels]

    return df[["epoch_index", "predicted_label", "predicted_name"]]
