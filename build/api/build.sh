#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

PROJECT_ROOT=$(git rev-parse --show-toplevel)
DOCKER_ROOT="$PROJECT_ROOT/build/api"
TMP_DIR="$DOCKER_ROOT/tmp"

AGENT_PACKAGE_ROOT="$PROJECT_ROOT/pkg/mancala_agent_pkg"
ENV_PACKAGE_ROOT="$PROJECT_ROOT/pkg/mancala_env_pkg"

# First build env whls
"$PROJECT_ROOT/build/env/build.sh"

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
mkdir -p "$TMP_DIR/deps"
mkdir -p "$TMP_DIR/run"

# TODO: this structure will definitely need to be changed
mkdir -p "$TMP_DIR/run/pkg/mancala_agent_pkg/inference_api"
mkdir -p "$TMP_DIR/run/pkg/mancala_agent_pkg/model"
mkdir -p "$TMP_DIR/run/saved_models/prod"

# Copy everything required to build dependencies into ./tmp/deps
cp "$AGENT_PACKAGE_ROOT/inference_api/inference_requirements.txt" "$TMP_DIR/deps/requirements.txt"
cp -r "$ENV_PACKAGE_ROOT/dist" "$TMP_DIR/deps/"

# Copy files required at run-time to ./tmp/run
cp "$AGENT_PACKAGE_ROOT/inference_api/"*.py "$TMP_DIR"/run/pkg/mancala_agent_pkg/inference_api/
cp "$AGENT_PACKAGE_ROOT/model/"*.py "$TMP_DIR"/run/pkg/mancala_agent_pkg/model/
cp -r "$PROJECT_ROOT/saved_models/prod" "$TMP_DIR/run/saved_models"

tree "$TMP_DIR"
(
	cd ./build/api || exit
	docker build --network=host --tag=mancala --file="$DOCKER_ROOT/Dockerfile" .
)
rm -rf "$TMP_DIR"
