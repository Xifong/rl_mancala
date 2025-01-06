#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

# IN:
#   - env var: export TARGET_VENV=<target_venv>
# ASSUMES:
#   - Python venv named 'venv' which has built requirements installed
#   - YMMV, probably swig installed on the system too
# SIDE-EFFECT: mancala_env package built into dist dir

PROJECT_ROOT=$(git rev-parse --show-toplevel)
PKG_NAME="mancala_env"
ENV_ROOT="$PROJECT_ROOT"/pkg/"$PKG_NAME"_pkg

# Activate development venv
source ./venv/bin/activate
(
	cd "$ENV_ROOT" || exit
	python -m build
)
