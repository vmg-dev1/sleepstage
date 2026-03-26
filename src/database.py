import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "db/sleepstage.db"

# Feature columns — must match exactly what features.py produces
FEATURE_COLS = [
    "delta_abs", "theta_abs", "alpha_abs", "sigma_abs", "beta_abs",
    "delta_rel", "theta_rel", "alpha_rel", "sigma_rel", "beta_rel",
    "eeg_variance", "zcr", "delta_beta_ratio",
    "eog_variance", "emg_variance",
]


def get_connection() -> sqlite3.Connection:
    """Open and return a SQLite connection to the database."""
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """
    Create all tables if they do not already exist.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.

    Tables
    ------
    epochs      : one row per 30s epoch, with features and ground truth label
    model_runs  : one row per training run, with hyperparameters and metrics
    """
    conn = get_connection()
    cursor = conn.cursor()

    # --- epochs table ---
    feature_cols_sql = "\n".join(
        [f"    {col} REAL NOT NULL," for col in FEATURE_COLS]
    )
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS epochs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id   TEXT NOT NULL,
            night        INTEGER NOT NULL,
            epoch_index  INTEGER NOT NULL,
            label        INTEGER NOT NULL,
            label_name   TEXT NOT NULL,
            {feature_cols_sql}
            created_at   TEXT NOT NULL
        )
    """)

    # Unique constraint: prevent duplicate extraction of the same epoch
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_epochs_unique
        ON epochs (patient_id, night, epoch_index)
    """)

    # Index for fast patient-level queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_epochs_patient
        ON epochs (patient_id)
    """)

    # --- model_runs table ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at      TEXT NOT NULL,
            algorithm       TEXT NOT NULL,
            params          TEXT,
            train_patients  TEXT NOT NULL,
            test_patients   TEXT NOT NULL,
            n_train_epochs  INTEGER,
            n_test_epochs   INTEGER,
            accuracy        REAL,
            f1_weighted     REAL,
            f1_W            REAL,
            f1_N1           REAL,
            f1_N2           REAL,
            f1_N3           REAL,
            f1_REM          REAL,
            notes           TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_features(df: pd.DataFrame, replace: bool = False) -> dict:
    """
    Write a features DataFrame to the epochs table.

    Parameters
    ----------
    df      : DataFrame produced by batch.run_batch_extraction()
    replace : if True, delete existing rows for the same patient_ids
              before inserting — useful to re-extract a patient.
              if False (default), skip rows that already exist (upsert).

    Returns
    -------
    dict with keys: inserted, skipped, patients
    """
    if df.empty:
        return {"inserted": 0, "skipped": 0, "patients": []}

    init_db()
    conn   = get_connection()
    cursor = conn.cursor()

    df = df.copy()
    df["created_at"] = datetime.now().isoformat()

    inserted = 0
    skipped  = 0
    patients = df["patient_id"].unique().tolist()

    if replace:
        for pid in patients:
            cursor.execute("DELETE FROM epochs WHERE patient_id = ?", (pid,))

    cols = ["patient_id", "night", "epoch_index", "label", "label_name"] \
       + FEATURE_COLS + ["created_at"]

    placeholders = ", ".join(["?"] * len(cols))
    col_names    = ", ".join(cols)

    for _, row in df.iterrows():
        values = tuple(row[c] for c in cols)
        try:
            cursor.execute(
                f"INSERT INTO epochs ({col_names}) VALUES ({placeholders})",
                values
            )
            inserted += 1
        except sqlite3.IntegrityError:
            # Duplicate (patient_id, epoch_index) — skip silently
            skipped += 1

    conn.commit()
    conn.close()

    return {"inserted": inserted, "skipped": skipped, "patients": patients}


def load_features(patient_ids: list = None) -> pd.DataFrame:
    """
    Load feature rows from the epochs table.

    Parameters
    ----------
    patient_ids : list of patient_id strings to load.
                  If None, loads all patients.

    Returns
    -------
    pd.DataFrame with all feature columns + patient_id, label, label_name
    """
    init_db()
    conn = get_connection()

    if patient_ids is None:
        df = pd.read_sql("SELECT * FROM epochs", conn)
    else:
        placeholders = ", ".join(["?"] * len(patient_ids))
        df = pd.read_sql(
            f"SELECT * FROM epochs WHERE patient_id IN ({placeholders})",
            conn,
            params=patient_ids
        )

    conn.close()
    return df


def get_patient_ids() -> list:
    """Return sorted list of all patient_ids currently in the database."""
    init_db()
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT patient_id FROM epochs ORDER BY patient_id")
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids


def get_db_summary() -> dict:
    """
    Return a summary of the current database contents.
    Useful for the dashboard to show what data is available.

    Returns
    -------
    dict with keys: n_patients, n_epochs, stage_counts, patients
    """
    init_db()
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(DISTINCT patient_id) FROM epochs")
    n_patients = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM epochs")
    n_epochs = cursor.fetchone()[0]

    cursor.execute("""
        SELECT label_name, COUNT(*) as count
        FROM epochs
        GROUP BY label_name
        ORDER BY label_name
    """)
    stage_counts = {row[0]: row[1] for row in cursor.fetchall()}

    conn.close()

    return {
        "n_patients":   n_patients,
        "n_epochs":     n_epochs,
        "stage_counts": stage_counts,
        "patients":     get_patient_ids(),
    }


def save_model_run(run: dict) -> int:
    """
    Save a model training run to the model_runs table.

    Parameters
    ----------
    run : dict with keys matching model_runs columns.
          train_patients and test_patients should be comma-separated strings.

    Returns
    -------
    int : id of the inserted row
    """
    init_db()
    conn   = get_connection()
    cursor = conn.cursor()

    run["trained_at"] = datetime.now().isoformat()

    cols         = list(run.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_names    = ", ".join(cols)
    values       = tuple(run[c] for c in cols)

    cursor.execute(
        f"INSERT INTO model_runs ({col_names}) VALUES ({placeholders})",
        values
    )
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return run_id


def load_model_runs() -> pd.DataFrame:
    """Load all model runs sorted by accuracy descending."""
    init_db()
    conn = get_connection()
    df   = pd.read_sql(
        "SELECT * FROM model_runs ORDER BY accuracy DESC",
        conn
    )
    conn.close()
    return df
