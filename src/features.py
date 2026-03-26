import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Standard EEG frequency bands used in sleep staging literature
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 15.0),
    "beta":  (15.0, 30.0),
}

EPOCH_DURATION = 30   # [sec] — standard polysomnography epoch length

STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # merged into N3 per modern AASM standard
    "Sleep stage R": 4,
    "Sleep stage M": -1,  # movement — excluded from training
}
STAGE_LABELS = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# Sleep-EDF SC* files: EEG and EOG at 100 Hz, EMG at 1 Hz.
# The EMG was hardware high-pass filtered, rectified and low-pass filtered
# before sampling — only the RMS envelope is stored, not the raw signal.
# ST* files have all channels at 100 Hz, but we use SC* for training.
SC_EMG_SFREQ = 1.0   # [Hz]


def _bandpower(psd, freqs, band):
    """
    Integrate PSD over a frequency band using the trapezoidal rule.
    Returns absolute power in µV².
    """
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    return float(np.trapz(psd[mask], freqs[mask]))


def _extract_epoch_features(epoch_eeg, epoch_eog, epoch_emg, sfreq):
    """
    Extract 15 features from a single 30s epoch.

    Parameters
    ----------
    epoch_eeg : np.ndarray  shape (sfreq*30,)    EEG Fpz-Cz in µV  — 100 Hz
    epoch_eog : np.ndarray  shape (sfreq*30,)    EOG horizontal in µV — 100 Hz
    epoch_emg : np.ndarray  shape (SC_EMG_SFREQ*30,)  EMG RMS envelope in µV — 1 Hz
    sfreq     : float       EEG/EOG sampling frequency in Hz (100 Hz for SC files)

    Returns
    -------
    dict of 15 float features
    """
    # --- EEG spectral features (Welch PSD) ---
    freqs, psd = welch(epoch_eeg, fs=sfreq, nperseg=int(sfreq * 4))

    abs_powers = {
        band: _bandpower(psd, freqs, frange)
        for band, frange in FREQ_BANDS.items()
    }
    total_power = sum(abs_powers.values()) + 1e-10  # avoid division by zero

    rel_powers = {
        f"{band}_rel": abs_powers[band] / total_power
        for band in FREQ_BANDS
    }
    abs_powers_renamed = {f"{band}_abs": v for band, v in abs_powers.items()}

    # --- EEG temporal features ---
    eeg_variance     = float(np.var(epoch_eeg))
    zero_crossings   = float(np.sum(np.diff(np.signbit(epoch_eeg))))
    zcr              = zero_crossings / (len(epoch_eeg) / sfreq)  # [crossings/sec]
    delta_beta_ratio = abs_powers["delta"] / (abs_powers["beta"] + 1e-10)

    # --- EOG feature ---
    # High variance during REM due to rapid eye movements
    eog_variance = float(np.var(epoch_eog))

    # --- EMG feature ---
    # EMG is the RMS envelope at 1 Hz — 30 samples per epoch for SC files.
    # Low variance during REM due to muscle atonia.
    # Variance is valid despite low sfreq — reflects slow amplitude changes.
    emg_variance = float(np.var(epoch_emg))

    return {
        **abs_powers_renamed,                  # delta_abs … beta_abs     (5)
        **rel_powers,                          # delta_rel … beta_rel     (5)
        "eeg_variance":      eeg_variance,     # µV²
        "zcr":               zcr,              # crossings/sec
        "delta_beta_ratio":  delta_beta_ratio, # dimensionless
        "eog_variance":      eog_variance,     # µV²
        "emg_variance":      emg_variance,     # µV²
    }                                          # total: 15 features


def extract_features(psg_path: str, hypnogram_path: str, patient_id: str) -> pd.DataFrame:
    """
    Extract features for all valid epochs in a PSG recording.

    Parameters
    ----------
    psg_path       : path to the PSG EDF file (SC* format assumed)
    hypnogram_path : path to the matching Hypnogram EDF file
    patient_id     : string identifier (e.g. "SC4001")

    Returns
    -------
    pd.DataFrame with columns:
        patient_id, epoch_index, label, label_name + 15 feature columns
    Epochs labelled -1 (movement) are excluded entirely.
    """
    raw   = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]   # EEG/EOG sampling frequency — 100 Hz for SC files

    # EEG and EOG: full 100 Hz resolution
    eeg = raw.get_data(picks=["EEG Fpz-Cz"])[0]    * 1e6  # V → µV
    eog = raw.get_data(picks=["EOG horizontal"])[0] * 1e6

    # EMG: 1 Hz RMS envelope for SC files — get_data returns it resampled
    # to match the EDF internal representation; we re-derive at 1 Hz below.
    # We read the raw channel and then downsample by taking every nth sample
    # to recover the original 1 Hz grid.
    emg_raw    = raw.get_data(picks=["EMG submental"])[0] * 1e6
    downsample = int(round(sfreq / SC_EMG_SFREQ))   # e.g. 100 for SC files
    emg        = emg_raw[::downsample]               # back to 1 Hz — shape: (n_sec,)

    annotations = mne.read_annotations(hypnogram_path)

    rows        = []
    epoch_index = 0

    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        label = STAGE_MAP.get(description, -1)
        if label == -1:
            epoch_index += 1
            continue

        n_sub_epochs = int(round(duration / EPOCH_DURATION))

        for i in range(n_sub_epochs):

            # EEG/EOG slice at sfreq (100 Hz)
            start_eeg = int((onset + i * EPOCH_DURATION) * sfreq)
            end_eeg   = start_eeg + int(EPOCH_DURATION * sfreq)

            # EMG slice at SC_EMG_SFREQ (1 Hz)
            start_emg = int((onset + i * EPOCH_DURATION) * SC_EMG_SFREQ)
            end_emg   = start_emg + int(EPOCH_DURATION * SC_EMG_SFREQ)

            # Guard: skip if either signal runs out of samples
            if end_eeg > eeg.shape[0] or end_emg > emg.shape[0]:
                break

            features = _extract_epoch_features(
                eeg[start_eeg:end_eeg],
                eog[start_eeg:end_eeg],
                emg[start_emg:end_emg],
                sfreq
            )

            rows.append({
                "patient_id":  patient_id,
                "epoch_index": epoch_index,
                "label":       label,
                "label_name":  STAGE_LABELS[label],
                **features
            })
            epoch_index += 1

    return pd.DataFrame(rows)
