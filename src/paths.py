"""Project paths, built from this file so they work on all our laptops (guidelines, code rules)."""
from pathlib import Path

# paths.py is in src/, so the root is one folder up
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# we create the folders here because data/ is in .gitignore, so it's not there after a clone
for _d in (DATA_DIR, MODEL_DIR, LOG_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# AI assistant used to double-check the code.
