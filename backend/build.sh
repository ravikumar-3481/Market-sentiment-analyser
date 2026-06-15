#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Optional: Add any extra build steps here (e.g., NLTK downloads, database migrations)
