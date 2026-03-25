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
    picks =mne.pick_channels(raw.info["ch.names"], include=[channel])
    data, times = raw[picks, :]
    signal = data[0]*1e6 #Volt to microVolt conversion
    return times, signal

def get_channel_names(raw):
    """
    Return a list of available channel names in the observed file.
    """
    return raw.info["ch_names"]
