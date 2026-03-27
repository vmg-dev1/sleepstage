#!/usr/bin/env bash
# =============================================================================
# download_patients.sh
# Download Sleep-EDF SC* recordings from PhysioNet.
#
# Strategy:
#   1. Fetch the full file listing from PhysioNet once (one HTTP request)
#   2. Build a local index of all available PSG and Hypnogram filenames
#   3. Download only the files matching the requested patient range
#
# Usage:
#   bash scripts/download_patients.sh [FIRST_PATIENT] [LAST_PATIENT]
#
# Examples:
#   bash scripts/download_patients.sh 0 8    # patients 00 to 08
#   bash scripts/download_patients.sh 4 9    # patients 04 to 09
#   bash scripts/download_patients.sh 0 78   # all patients
#
# Compatible with macOS (curl only, no wget).
# =============================================================================

BASE_URL="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
DATA_DIR="data"

FIRST=$(( 10#${1:-0} ))
LAST=$(( 10#${2:-8} ))

echo "============================================="
echo "Sleep-EDF downloader (macOS)"
echo "Patients: $(printf '%02d' $FIRST) → $(printf '%02d' $LAST)"
echo "Destination: $DATA_DIR/"
echo "============================================="

mkdir -p "$DATA_DIR"

# --- Step 1: Fetch directory listing once ---
echo ""
echo "Fetching file index from PhysioNet..."
LISTING=$(curl -s "${BASE_URL}/")

if [ -z "$LISTING" ]; then
    echo "ERROR: could not fetch directory listing from PhysioNet."
    exit 1
fi

# Extract all .edf filenames from the HTML listing
ALL_FILES=$(echo "$LISTING" | grep -o 'SC4[^"]*\.edf' | sort -u)
echo "Found $(echo "$ALL_FILES" | wc -l | tr -d ' ') EDF files in index."

n_downloaded=0
n_skipped=0
n_failed=0

# --- Step 2: Loop over requested patients ---
for (( i=FIRST; i<=LAST; i++ )); do
    pad=$(printf "%02d" "$i")
    echo ""
    echo "--- Patient $pad ---"

    night=1
    found_any=false

    while true; do
        prefix="SC4${pad}${night}"

        # Find PSG file for this patient/night
        psg_file=$(echo "$ALL_FILES" | grep "^${prefix}E0-PSG")
        # Find Hypnogram file for this patient/night (any letter)
        hyp_file=$(echo "$ALL_FILES" | grep "^${prefix}E.-Hypnogram")

        if [ -z "$psg_file" ]; then
            if [ "$found_any" = false ]; then
                echo "  No recordings found for patient $pad"
                n_failed=$((n_failed + 1))
            else
                echo "  No more nights after night $((night - 1))"
            fi
            break
        fi

        found_any=true
        echo "  Night $night:"
        echo "    PSG:       $psg_file"
        echo "    Hypnogram: ${hyp_file:-NOT FOUND IN INDEX}"

        # --- Download PSG ---
        psg_dest="${DATA_DIR}/${psg_file}"
        if [ -f "$psg_dest" ]; then
            echo "    PSG: already exists, skipped"
            n_skipped=$((n_skipped + 1))
        else
            echo -n "    PSG: downloading..."
            curl -# -o "$psg_dest" "${BASE_URL}/${psg_file}" && \
                echo " done" || echo " FAILED"
            n_downloaded=$((n_downloaded + 1))
        fi

        # --- Download Hypnogram ---
        if [ -z "$hyp_file" ]; then
            echo "    Hypnogram: not found in index, skipping"
        else
            hyp_dest="${DATA_DIR}/${hyp_file}"
            if [ -f "$hyp_dest" ]; then
                echo "    Hypnogram: already exists, skipped"
                n_skipped=$((n_skipped + 1))
            else
                echo -n "    Hypnogram: downloading..."
                curl -s -o "$hyp_dest" "${BASE_URL}/${hyp_file}" && \
                    echo " done" || echo " FAILED"
                n_downloaded=$((n_downloaded + 1))
            fi
        fi

        night=$((night + 1))
    done
done

echo ""
echo "============================================="
echo "Done."
echo "  Files downloaded  : $n_downloaded"
echo "  Files skipped     : $n_skipped"
echo "  Patients not found: $n_failed"
echo "============================================="
echo ""
echo "Run feature extraction to update the database:"
echo "  python -c \"from src.batch import run_batch_extraction; from src.database import save_features; save_features(run_batch_extraction())\""
