import os
import re
import logging
import pandas as pd
from datetime import datetime
from src.features import extract_features

# --- Logging setup ---
# Log file lives in db/ and is excluded from git via .gitignore
os.makedirs("db", exist_ok=True)
log_path = os.path.join("db", "extraction.log")

logging.basicConfig(
    filename=log_path,
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def discover_recordings(data_dir: str = "data/") -> pd.DataFrame:
    """
    Scan data_dir and build a DataFrame with one row per valid PSG recording.

    A valid recording requires:
      - a PSG file  matching pattern *PSG*.edf
      - a matching Hypnogram file in the same directory

    The patient_id is extracted from the filename using the Sleep-EDF
    naming convention: SC4001E0-PSG.edf → patient_id = "SC4001"

    Returns
    -------
    pd.DataFrame with columns:
        patient_id | psg_path | hypnogram_path | has_hypnogram
    """
    records = []

    all_files = [f for f in os.listdir(data_dir) if f.endswith(".edf")]
    psg_files = [f for f in all_files if "PSG" in f]

    for psg_file in sorted(psg_files):
        psg_path = os.path.join(data_dir, psg_file)

        # Extract patient_id from filename (e.g. SC4001 from SC4001E0-PSG.edf)
        match = re.match(r"^([A-Z]{2}\d{4})", psg_file)
        patient_id = match.group(1) if match else psg_file.replace("-PSG.edf", "")

        # Find matching hypnogram (same prefix, contains "Hypnogram")
        hypnogram_file = psg_file.replace("E0-PSG", "EC-Hypnogram")
        hypnogram_path = os.path.join(data_dir, hypnogram_file)
        has_hypnogram  = os.path.exists(hypnogram_path)

        if not has_hypnogram:
            hypnogram_path = None

        records.append({
            "patient_id":     patient_id,
            "psg_path":       psg_path,
            "hypnogram_path": hypnogram_path,
            "has_hypnogram":  has_hypnogram,
        })

    df = pd.DataFrame(records)
    logger.info(f"Discovery complete — {len(df)} PSG files found, "
                f"{df['has_hypnogram'].sum()} with matching hypnogram.")
    return df


def run_batch_extraction(data_dir: str = "data/") -> pd.DataFrame:
    """
    Run feature extraction for all valid recordings in data_dir.

    Skips recordings with missing hypnogram or any extraction error.
    Errors are logged to db/extraction.log — partial data is never written.

    Returns
    -------
    pd.DataFrame with all features for all valid epochs across all recordings.
    An empty DataFrame is returned if no valid recordings are found.
    """
    recordings = discover_recordings(data_dir)
    all_features = []
    n_success = 0
    n_skip    = 0
    n_error   = 0

    logger.info(f"Starting batch extraction — {datetime.now().isoformat()}")

    for _, row in recordings.iterrows():
        patient_id = row["patient_id"]
        psg_path   = row["psg_path"]

        # --- Guard: missing hypnogram ---
        if not row["has_hypnogram"]:
            msg = (f"SKIP | {patient_id} | {os.path.basename(psg_path)} | "
                   f"no matching hypnogram found")
            logger.warning(msg)
            n_skip += 1
            continue

        hypnogram_path = row["hypnogram_path"]

        # --- Extract features with full error isolation ---
        # If anything fails for this recording, we log and skip entirely.
        # No partial data is appended to all_features.
        try:
            df_patient = extract_features(psg_path, hypnogram_path, patient_id)

            if df_patient.empty:
                msg = (f"SKIP | {patient_id} | {os.path.basename(psg_path)} | "
                       f"extraction returned empty DataFrame")
                logger.warning(msg)
                n_skip += 1
                continue

            n_epochs = len(df_patient)
            stage_counts = df_patient["label_name"].value_counts().to_dict()
            logger.info(
                f"OK | {patient_id} | {os.path.basename(psg_path)} | "
                f"{n_epochs} epochs | {stage_counts}"
            )
            all_features.append(df_patient)
            n_success += 1

        except Exception as e:
            msg = (f"ERROR | {patient_id} | {os.path.basename(psg_path)} | "
                   f"{type(e).__name__}: {e}")
            logger.error(msg)
            n_error += 1
            # Do not append anything — this recording is fully discarded
            continue

    # --- Summary ---
    logger.info(
        f"Batch complete — success: {n_success} | "
        f"skipped: {n_skip} | errors: {n_error}"
    )

    if not all_features:
        logger.warning("No features extracted — returning empty DataFrame.")
        return pd.DataFrame()

    return pd.concat(all_features, ignore_index=True)
