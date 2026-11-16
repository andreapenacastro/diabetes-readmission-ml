"""Project paths, resolved from this file's location so they work on any machine."""
from pathlib import Path

# This file lives in src/, so the project root is one folder up
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

for _d in (DATA_DIR, MODEL_DIR, LOG_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
