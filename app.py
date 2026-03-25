import streamlit as st
import plotly.graph_objects as go
from src.edf_loader import load_edf, get_signal_dataframe, get_channel_names

st.set_page_config(page_title="SleepStage Analyser", layout="wide")
st.title("SleepStage Analyser")
st.caption("Automatic sleep stage classification from EEG/EOG/EMG signals")

# --- File selection ---
st.subheader("Load a recording")
uploaded_file = st.file_uploader("Select a PSG EDF file", type=["edf"])

if uploaded_file is not None:
    # Save uploaded file temporarily to disk (MNE needs a real path)
    tmp_path = f"data/tmp_{uploaded_file.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    raw = load_edf(tmp_path)
    channels = get_channel_names(raw)

    # --- Channel selector ---
    selected_channel = st.selectbox("Select channel to display", channels)
    times, signal = get_signal_dataframe(raw, channel=selected_channel)

    # --- Time window slider ---
    st.subheader("Raw EEG signal")
    max_time = int(times[-1])
    window_start = st.slider("Start time (seconds)", 0, max(0, max_time - 30), 0)
    window_end = window_start + 30

    # Filter to selected window
    mask = (times >= window_start) & (times <= window_end)
    t_window = times[mask]
    s_window = signal[mask]

    # --- Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_window,
        y=s_window,
        mode="lines",
        line=dict(width=0.8, color="#7F77DD"),
        name=selected_channel
    ))
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        height=300,
        margin=dict(l=0, r=0, t=10, b=40),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Recording duration: {max_time // 60} min {max_time % 60} s — "
               f"Sampling rate: {int(raw.info['sfreq'])} Hz — "
               f"Channels: {len(channels)}")
else:
    st.info("Upload a PSG EDF file to get started.")
    