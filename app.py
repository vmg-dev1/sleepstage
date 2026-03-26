import os
import streamlit as st
import plotly.graph_objects as go
from src.edf_loader import load_edf, get_signal_dataframe, get_channel_names, load_annotations

st.set_page_config(page_title="SleepStage Analyser", layout="wide")

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet">
<style>
    :root {
        --ink:          #1a1a1a;
        --ink-muted:    #4a4a4a;
        --bg:           #f7f5f0;
        --bg-card:      #ffffff;
        --accent:       #8297a8;
        --accent-light: #e8eef8;
        --border:       #e2ddd6;
        --serif:        'DM Serif Display', Georgia, serif;
        --sans:         'DM Sans', sans-serif;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: var(--bg) !important;
        color: var(--ink) !important;
        font-family: var(--sans) !important;
        font-size: 18px !important;
    }
    [data-testid="stSidebar"] {
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
        font-family: var(--sans) !important;
    }
    [data-testid="stSidebarNav"] a,
    [data-testid="stSidebarNav"] span,
    [data-testid="stSidebarNavItems"] a,
    [data-testid="stSidebarNavItems"] span,
    [data-testid="stSidebarNavLink"] span {
        color: var(--ink) !important;
        font-family: var(--sans) !important;
        font-size: 15px !important;
    }
    [data-testid="stSidebarNavLink"]:hover,
    [data-testid="stSidebarNavLink"][aria-selected="true"] {
        background-color: var(--accent-light) !important;
        border-radius: 6px !important;
    }
    h1, h2, h3, .stTitle {
        font-family: var(--serif) !important;
        color: var(--ink) !important;
        font-weight: 400 !important;
    }
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-secondary"] p,
    [data-testid="stBaseButton-secondary"] span {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--ink) !important;
        font-family: var(--sans) !important;
        font-size: 14px !important;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        background-color: var(--accent-light) !important;
        border-color: var(--accent) !important;
    }
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"] p,
    [data-testid="stBaseButton-primary"] span {
        background-color: #7F77DD !important;
        border: none !important;
        color: #ffffff !important;
        font-family: var(--sans) !important;
        font-size: 14px !important;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #534AB7 !important;
    }
    .stSelectbox label, .stSlider label, .stRadio label,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p {
        font-family: var(--sans) !important;
        font-size: 15px !important;
        color: var(--ink) !important;
        font-weight: 500 !important;
    }
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p,
    [data-testid="stRadio"] span,
    [data-testid="stRadio"] div {
        color: var(--ink) !important;
        font-family: var(--sans) !important;
        font-size: 15px !important;
    }
    [data-testid="stSelectbox"] > div > div {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        color: var(--ink) !important;
        font-family: var(--sans) !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--accent) !important;
    }
    /* Align all sliders with the Plotly plot area (compensates for l=50 margin) */
    [data-testid="stSlider"] {
        padding-left: 50px !important;
        padding-right: 10px !important;
    }
    [data-testid="stAlert"],
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div {
        background-color: var(--accent-light) !important;
        border: 1px solid var(--border) !important;
        color: var(--ink) !important;
        font-family: var(--sans) !important;
    }
    [data-testid="stPlotlyChart"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 8px !important;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span {
        color: var(--ink) !important;
        font-family: var(--sans) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("SleepStage Analyser")
st.caption("Automatic sleep stage classification from EEG/EOG/EMG signals")

STAGE_COLORS = {
    "W":   "#888780",
    "N1":  "#85B7EB",
    "N2":  "#378ADD",
    "N3":  "#7F77DD",
    "REM": "#EF9F27",
    "M":   "#F09595",
    "?":   "#444441"
}
STAGE_ORDER = {"W": 5, "REM": 4, "N1": 3, "N2": 2, "N3": 1, "M": 0}
STAGE_LABELS = {
    "W":   "Eveil (Awake)",
    "N1":  "Light sleep — Stage 1",
    "N2":  "Light sleep — Stage 2",
    "N3":  "Deep sleep — Stage 3 & 4",
    "REM": "Rapid Eye Movement",
    "M":   "Movement time",
    "?":   "Unknown",
}

MAX_WINDOW_SEC = 300

# --- Legend ---
legend_html = '<div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:20px;">'
for stage, label in STAGE_LABELS.items():
    color = STAGE_COLORS[stage]
    legend_html += (
        f'<div style="display:flex; align-items:center; gap:8px; '
        f'background:#ffffff; border:1px solid #e2ddd6; border-radius:8px; padding:6px 14px;">'
        f'<span style="display:inline-block; width:12px; height:12px; '
        f'border-radius:3px; background:{color};"></span>'
        f'<span style="font-size:14px; color:#1a1a1a; font-family:\'DM Sans\', sans-serif;">'
        f'<b>{stage}</b> — {label}</span>'
        f'</div>'
    )
legend_html += '</div>'
st.markdown(legend_html, unsafe_allow_html=True)

# --- File selection ---
st.subheader("Load a recording")
psg_files = [f for f in os.listdir("data/") if f.endswith(".edf") and "PSG" in f]
psg_files.sort()

if len(psg_files) == 0:
    st.warning("No PSG EDF files found in the data/ folder.")
    st.stop()

selected_file = st.selectbox("Select a PSG recording", psg_files)
raw           = load_edf(f"data/{selected_file}")
channels      = get_channel_names(raw)

hypnogram_file = selected_file.replace("E0-PSG", "EC-Hypnogram")
hypnogram_path = f"data/{hypnogram_file}"
annotations    = load_annotations(hypnogram_path) if os.path.exists(hypnogram_path) else []

selected_channel = st.selectbox("Select channel to display", channels)
times, signal    = get_signal_dataframe(raw, channel=selected_channel)
max_time         = int(times[-1])
max_min          = max_time / 60

# --- Session state init ---
if "zoom_level" not in st.session_state:
    st.session_state.zoom_level = 0
if "scroll_pos" not in st.session_state:
    st.session_state.scroll_pos = 0.0
if "cursor_min" not in st.session_state:
    st.session_state.cursor_min = 0.0

# --- Link mode ---
link_mode = st.radio(
    "Graph linking",
    options=["Unlinked — use position slider", "Linked — select region on hypnogram"],
    horizontal=True,
    index=0
)
linked = link_mode == "Linked — select region on hypnogram"

# --- Hypnogram section ---
st.subheader("Hypnogram — ground truth")

# Zoom slider
st.caption("Zoom level")
zoom_level = st.slider(
    "Zoom level",
    min_value=0,
    max_value=95,
    value=st.session_state.zoom_level,
    step=5,
    label_visibility="collapsed"
)
st.session_state.zoom_level = zoom_level

visible_fraction = 1.0 - (zoom_level / 100.0)
visible_duration = max_min * visible_fraction
is_zoomed        = zoom_level > 0

# Scroll slider
st.caption("Scroll")
if is_zoomed:
    max_scroll = max_min - visible_duration
    scroll_pos = st.slider(
        "Scroll position",
        min_value=0.0,
        max_value=float(max_scroll),
        value=float(min(st.session_state.scroll_pos, max_scroll)),
        step=float(visible_duration / 20),
        format="%.1f min",
        label_visibility="collapsed",
        key="scroll_slider"
    )
    st.session_state.scroll_pos = scroll_pos
else:
    st.slider(
        "Scroll position",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        disabled=True,
        label_visibility="collapsed",
        key="scroll_slider_disabled"
    )
    scroll_pos = 0.0

x_start = scroll_pos
x_end   = scroll_pos + visible_duration

# Position slider — spans the visible window, controls red cursor
if not linked:
    prev_cursor = st.session_state.cursor_min
    clamped     = float(max(x_start, min(x_end, prev_cursor)))

    st.caption("Position")
    cursor_min = st.slider(
        "Signal position",
        min_value=float(x_start),
        max_value=float(x_end),
        value=clamped,
        step=float(max(0.05, visible_duration / 500)),
        format="%.1f min",
        label_visibility="collapsed",
        key="cursor_slider"
    )
    st.session_state.cursor_min = cursor_min

    sig_start = max(0.0, cursor_min * 60 - 15)
    sig_end   = min(float(max_time), cursor_min * 60 + 15)
else:
    cursor_min = None
    sig_start  = 0.0
    sig_end    = 30.0

# --- Build hypnogram figure ---
hyp_x, hyp_y = [], []
for onset, duration, label in annotations:
    if label not in STAGE_ORDER:
        continue
    hyp_x += [onset / 60, (onset + duration) / 60]
    hyp_y += [STAGE_ORDER[label], STAGE_ORDER[label]]

fig_hyp = go.Figure()

# Colored background bands
for onset, duration, label in annotations:
    if label not in STAGE_ORDER:
        continue
    fig_hyp.add_shape(
        type="rect",
        x0=onset / 60,
        x1=(onset + duration) / 60,
        y0=STAGE_ORDER[label] - 0.4,
        y1=STAGE_ORDER[label] + 0.4,
        fillcolor=STAGE_COLORS.get(label, "#444441"),
        opacity=0.3,
        line_width=0,
    )

# Red cursor line aligned with position slider
if not linked and cursor_min is not None:
    fig_hyp.add_shape(
        type="line",
        x0=cursor_min,
        x1=cursor_min,
        y0=-0.8,
        y1=5.8,
        line=dict(color="#e05c2a", width=2, dash="dot")
    )
    fig_hyp.add_annotation(
        x=cursor_min,
        y=5.8,
        text=f"{cursor_min:.1f}m",
        showarrow=False,
        font=dict(size=11, color="#e05c2a", family="DM Sans"),
        bgcolor="#ffffff",
        borderpad=2,
        yanchor="bottom",
    )

# Hypnogram line
fig_hyp.add_trace(go.Scatter(
    x=hyp_x,
    y=hyp_y,
    mode="lines",
    line=dict(width=2, color="#7F77DD"),
    name="Ground truth",
))

fig_hyp.update_layout(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#1a1a1a", family="DM Sans", size=14),
    xaxis=dict(
        title=dict(text="Time (minutes)", font=dict(color="#4a4a4a", size=14)),
        range=[x_start, x_end],
        gridcolor="#e2ddd6",
        linecolor="#e2ddd6",
        color="#1a1a1a",
        tickfont=dict(color="#1a1a1a", size=13),
        fixedrange=True,
    ),
    yaxis=dict(
        tickvals=list(STAGE_ORDER.values()),
        ticktext=list(STAGE_ORDER.keys()),
        range=[-0.8, 5.8],
        gridcolor="#e2ddd6",
        linecolor="#e2ddd6",
        color="#1a1a1a",
        tickfont=dict(color="#1a1a1a", size=13),
        fixedrange=True,
    ),
    height=220,
    margin=dict(l=50, r=10, t=10, b=40),
    dragmode="select" if linked else False,
    selectdirection="h",
)

hyp_event = st.plotly_chart(
    fig_hyp,
    use_container_width=True,
    on_select="rerun",
    key="hypnogram"
)

# Reset + info
col_reset, col_info = st.columns([2, 13])
with col_reset:
    if st.button("Reset view", help="Reset zoom and scroll to full recording"):
        st.session_state.zoom_level = 0
        st.session_state.scroll_pos = 0.0
        st.session_state.cursor_min = 0.0
        st.rerun()
with col_info:
    if is_zoomed:
        st.caption(f"Viewing {visible_duration:.0f} min window — scroll to navigate")
    else:
        st.caption("Full recording view")

# --- Linked mode: determine signal window from hypnogram box selection ---
if linked:
    st.caption("Draw a horizontal selection on the hypnogram. Max 5 minutes.")
    
    sig_start = None
    sig_end   = None

    try:
        box = hyp_event.selection.get("box", [])
        if box and len(box) > 0:
            x_range = box[0].get("x", [])
            if len(x_range) == 2:
                sig_start = float(x_range[0]) * 60
                sig_end   = float(x_range[1]) * 60
    except (KeyError, TypeError, AttributeError):
        pass

    if sig_start is None:
        st.info("Select a region on the hypnogram to display the raw signal.")
        st.stop()

    if (sig_end - sig_start) > MAX_WINDOW_SEC:
        st.warning(
            f"Selection too wide ({int(sig_end - sig_start)}s) — "
            f"please select less than {MAX_WINDOW_SEC // 60} minutes."
        )
        st.stop()

# --- Raw signal ---
st.subheader("Raw EEG signal")

mask     = (times >= sig_start) & (times <= sig_end)
t_window = times[mask]
s_window = signal[mask]

current_stage = "?"
for onset, duration, label in annotations:
    if onset <= sig_start < onset + duration:
        current_stage = label
        break

fig_sig = go.Figure()
fig_sig.add_trace(go.Scatter(
    x=t_window,
    y=s_window,
    mode="lines",
    line=dict(width=1.5, color=STAGE_COLORS.get(current_stage, "#7F77DD")),
    name=selected_channel
))
fig_sig.update_layout(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#1a1a1a", family="DM Sans", size=14),
    xaxis=dict(
        title=dict(text="Time (s)", font=dict(color="#4a4a4a", size=14)),
        gridcolor="#e2ddd6",
        linecolor="#e2ddd6",
        color="#1a1a1a",
        tickfont=dict(color="#1a1a1a", size=13),
    ),
    yaxis=dict(
        title=dict(text="Amplitude (µV)", font=dict(color="#4a4a4a", size=14)),
        gridcolor="#e2ddd6",
        linecolor="#e2ddd6",
        color="#1a1a1a",
        tickfont=dict(color="#1a1a1a", size=13),
    ),
    height=300,
    margin=dict(l=50, r=10, t=10, b=40),
    hovermode="x unified"
)
st.plotly_chart(fig_sig, use_container_width=True)

# --- Stage badge ---
color = STAGE_COLORS.get(current_stage, "#444441")
st.markdown(
    f'<div style="display:inline-block; background:{color}; color:white; '
    f'padding:6px 18px; border-radius:8px; font-weight:500; font-size:15px; '
    f'font-family:\'DM Sans\', sans-serif;">'
    f'Ground truth: {current_stage}</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Recording duration: {max_time // 60} min {max_time % 60} s — "
    f"Sampling rate: {int(raw.info['sfreq'])} Hz — "
    f"Channels: {len(channels)}"
)
