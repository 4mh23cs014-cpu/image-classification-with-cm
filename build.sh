#!/usr/bin/env bash
set -o errexit

# Upgrade packaging tools first
pip install --upgrade pip setuptools wheel

# Install a compatible numpy wheel first to satisfy scikit-learn build deps
pip install --upgrade "numpy==1.24.3"

# Then install remaining requirements
pip install -r requirements.txt
