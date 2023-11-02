#!/usr/bin/env bash
set -e

echo "Creating virtual environment"
python3.7 -m venv hps-env
echo "Activating virtual environment"

source $PWD/hps-env/bin/activate

$PWD/hps-env/bin/pip install -r requirements.txt