#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -u

echo "Starting build script..."
# Use python -m pip to avoid ambiguous pip executables
python -m pip install --upgrade pip setuptools wheel

# Install a compatible numpy wheel first to satisfy scikit-learn build deps
python -m pip install --upgrade "numpy==1.24.3"

# Then install remaining requirements; prefer binary wheels to avoid building from source where possible
python -m pip install --prefer-binary -v -r requirements.txt

echo "Build script completed successfully."
