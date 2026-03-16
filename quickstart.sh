#!/usr/bin/env bash
# =============================================================================
# Flood Risk Mapping — Quick Start Script
# Usage: bash quickstart.sh [demo|train|predict|dashboard]
# =============================================================================

set -e
cd "$(dirname "$0")"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

banner() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   🛰️  Flood Risk Mapping System                  ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

step() { echo -e "${GREEN}▶ $1${NC}"; }
info() { echo -e "${YELLOW}  ℹ  $1${NC}"; }
err()  { echo -e "${RED}  ✗  $1${NC}"; exit 1; }

banner

MODE=${1:-demo}

# ── Check Python ─────────────────────────────────────────────────────────
python3 --version > /dev/null 2>&1 || err "Python 3 not found."

# ── Virtual environment ───────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    step "Creating virtual environment..."
    python3 -m venv venv
fi

step "Activating virtual environment..."
source venv/bin/activate

# ── Install dependencies ──────────────────────────────────────────────────
if [ ! -f "venv/.installed" ]; then
    step "Installing dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    touch venv/.installed
    echo -e "${GREEN}  Dependencies installed successfully.${NC}"
else
    info "Dependencies already installed."
fi

echo ""
echo -e "Mode: ${YELLOW}${MODE}${NC}"
echo ""

case $MODE in
    demo)
        step "Generating synthetic demo data..."
        python3 src/utils/generate_demo_data.py --n 120

        step "Training Vision Transformer (demo — 5 epochs)..."
        python3 train.py --model vit --epochs 5 --prepare

        step "Training CNN baseline (demo — 5 epochs)..."
        python3 train.py --model cnn --epochs 5

        step "Running model comparison..."
        python3 evaluate.py --compare

        step "Launching dashboard..."
        info "Dashboard will open at http://localhost:8501"
        streamlit run dashboard/app.py
        ;;

    train)
        step "Full training pipeline..."
        python3 train.py --model vit --demo
        python3 train.py --model cnn
        python3 evaluate.py --compare
        ;;

    predict)
        IMG=${2:-""}
        if [ -z "$IMG" ]; then
            # Find any sample image
            IMG=$(find data -name "*.jpg" | head -1)
        fi
        if [ -z "$IMG" ]; then
            err "No image found. Provide: bash quickstart.sh predict <path/to/image.jpg>"
        fi
        step "Running prediction on: $IMG"
        python3 predict.py --image "$IMG" --model both --visualize
        ;;

    dashboard)
        step "Launching dashboard..."
        info "Open http://localhost:8501 in your browser"
        streamlit run dashboard/app.py
        ;;

    install)
        step "Installation complete."
        ;;

    *)
        echo "Usage: bash quickstart.sh [demo|train|predict|dashboard|install]"
        echo ""
        echo "  demo       — Generate data, train (5 epochs), launch dashboard"
        echo "  train      — Full training (30 epochs)"
        echo "  predict    — Run prediction on an image"
        echo "  dashboard  — Launch Streamlit dashboard"
        echo "  install    — Install dependencies only"
        ;;
esac
