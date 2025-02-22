#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

# IN:
#   - env var: export TARGET_VENV=<target_venv>
# ASSUMES:
#   - Python venv named 'venv' which has built requirements installed
#   - YMMV, probably swig installed on the system too
# SIDE-EFFECT: mancala_env package installed into target venv

# TODO: deduplicate this from the headless build.sh
PROJECT_ROOT=$(git rev-parse --show-toplevel)
PKG_NAME="mancala_env"
ENV_ROOT="$PROJECT_ROOT"/pkg/"$PKG_NAME"_pkg

# Activate development venv
source ./venv/bin/activate
(
	cd "$ENV_ROOT" || exit
	python -m build
)
source "./$TARGET_VENV/bin/activate"
pip uninstall "$PKG_NAME" -y
# TODO: parameterise this package version
pip install --no-cache-dir "$ENV_ROOT/dist/$PKG_NAME-0.0.1-py3-none-any.whl"
