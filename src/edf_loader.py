import mne
import numpy as np
# mne stands for magnetoeencephalography and Electroencephalography -> here this is the open source library developped by the MIT.
# psg stands for Polysomnography: medical term for full sleep recording (EEG + ECG + EMG)

def load_edf(psg_path: str):
    """
    Load a PSG EDF file and return raw data with chanel info.
    return the MNE Raw object.
    """
    raw = mne.io.read_raw_edf(psg_path, preload=True,verbose=False)
    return raw

def get_signal_dataframe(raw, channel: str = "EEG Fpz-Cz"):
    """
    Extract a single channel as a time/value numpy array pairs.
    Return times [sec] amd sognal [muV]
    """
    picks =mne.pick_channels(raw.info["ch_names"], include=[channel])
    data, times = raw[picks, :]
    signal = data[0]*1e6 #Volt to microVolt conversion
    return times, signal

def get_channel_names(raw):
    """
    Return a list of available channel names in the observed file.
    """
    return raw.info["ch_names"]

def load_annotations(hypnogram_path: str):
    """
    Load sleep stage annotations from a hypnogram EDF file.
    Returns a list of (onset_sec, duration_sec, stage_label) tuples.
    Sleep-EDF uses old AASM labels: W, R, 1, 2, 3, 4, M
    We map them to modern standard: W, REM, N1, N2, N3, N3, M
    """
    STAGE_MAP = {
        "Sleep stage W": "W",
        "Sleep stage R": "REM",
        "Sleep stage 1": "N1",
        "Sleep stage 2": "N2",
        "Sleep stage 3": "N3",
        "Sleep stage 4": "N3",  # merged with N3 per modern AASM standard
        "Sleep stage M": "M",
        "Movement time": "M",
    }

    annotations = mne.read_annotations(hypnogram_path)
    stages = []
    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description
    ):
        label = STAGE_MAP.get(description, "?")
        stages.append((float(onset), float(duration), label))
    return stages

