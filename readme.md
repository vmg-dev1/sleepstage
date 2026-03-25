# SleepStage Analyser

Interactive dashboard for automatic sleep stage classification
from EEG/EOG/EMG signals.

**Stack**: Python · Streamlit · MNE-Python · scikit-learn · SQLite · Plotly

## Setup

### 1. Clone the repo
git clone https://github.com/vmg-dev1/sleepstage.git
cd sleepstage

### 2. Virtual environment
python -m venv venv
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Data
Download EDF files from:
https://physionet.org/content/sleep-edfx/1.0.0/
Place them in the `data/` folder.

### 5. Run
streamlit run app.py

## Dataset
Sleep-EDF Database Expanded — Kemp et al., Physionet
78 healthy subjects, 2 nights each, annotated by sleep experts.
Signals: EEG (Fpz-Cz, Pz-Oz), EOG, EMG submental.

## Sleep stages
| Stage | Description |
|-------|-------------|
| W | Wake |
| N1 | Light sleep (transition) |
| N2 | Light sleep (spindles) |
| N3 | Deep sleep (slow waves) |
| REM | Rapid Eye Movement |

## Project structure
```
sleepstage/
├── data/          # EDF files (not versioned)
├── db/            # SQLite database (not versioned)
├── model/         # Trained models
├── pages/         # Streamlit pages 2-4
├── src/           # Core logic (loader, features, db, training)
├── app.py         # Main Streamlit page
└── requirements.txt
```

## License
MIT
