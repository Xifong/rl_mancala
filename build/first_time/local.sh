#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

PROJECT_ROOT=$(git rev-parse --show-toplevel)
VENV="venv"
INFERENCE_VENV="inference_venv"

# Create development venv (includes Torch CUDA)
python3 -m venv "$VENV"
source "$PROJECT_ROOT/$VENV/bin/activate"
pip install -r "$PROJECT_ROOT/development_requirements.txt"

# Create inference venv (removes Torch CUDA)
python3 -m venv "$INFERENCE_VENV"
source "$PROJECT_ROOT/$INFERENCE_VENV/bin/activate"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r "$PROJECT_ROOT/pkg/mancala_agent_pkg/inference_api/inference_requirements.txt"

# Switch back to development venv
source "$PROJECT_ROOT/$VENV/bin/activate"
