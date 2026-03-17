#!/usr/bin/env bash
set -euo pipefail

# Download dataset
huggingface-cli download torchgeo/ChesapeakeRSC --repo-type dataset --local-dir data/

# Download pretrained models
huggingface-cli download isaaccorley/chesapeakersc --local-dir models/
