# UrbanFloodNet Data Configuration
# ==================================
# Loads settings from configs/data.yaml and exposes them as module-level constants.
# Environment variable SELECTED_MODEL overrides the YAML value.

import os
from pathlib import Path

import yaml

# Locate project root and config file
_ROOT_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT_DIR / "configs" / "data.yaml"

with open(_CONFIG_PATH, "r") as f:
    _cfg = yaml.safe_load(f)

# ===== Exported constants (same names used throughout the codebase) =====
SELECTED_MODEL = os.environ.get("SELECTED_MODEL", _cfg["selected_model"])
DATA_FOLDER = _cfg["data_folder"]
MAX_EVENTS = _cfg["max_events"]
VALIDATION_SPLIT = _cfg["validation_split"]
TEST_SPLIT = _cfg["test_split"]
RANDOM_SEED = _cfg["random_seed"]

# ===== Derived paths =====
BASE_PATH = f"{DATA_FOLDER}/{SELECTED_MODEL}"
TRAIN_PATH = f"{BASE_PATH}/train"


# ===== Path validation =====
def validate_data_paths():
    """Validate that all required data files exist."""
    required_files = [
        f"{TRAIN_PATH}/1d_nodes_static.csv",
        f"{TRAIN_PATH}/2d_nodes_static.csv",
        f"{TRAIN_PATH}/1d_edges_static.csv",
        f"{TRAIN_PATH}/2d_edges_static.csv",
        f"{TRAIN_PATH}/1d_edge_index.csv",
        f"{TRAIN_PATH}/2d_edge_index.csv",
        f"{TRAIN_PATH}/1d2d_connections.csv",
    ]

    missing = []
    for fpath in required_files:
        if not os.path.exists(fpath):
            missing.append(fpath)

    if missing:
        print("\n[ERROR] Missing required data files:")
        for fpath in missing:
            print(f"  - {fpath}")
        print(f"\nExpected structure:")
        print(f"  UrbanFloodNet/data/{SELECTED_MODEL}/train/")
        raise FileNotFoundError(
            f"Data validation failed for {SELECTED_MODEL}. "
            f"See paths above and ensure data is in the correct location."
        )

    # Check for event directories
    event_dirs = [d for d in os.listdir(TRAIN_PATH) if d.startswith("event_")]
    if not event_dirs:
        print(f"\n[WARNING] No event directories found in {TRAIN_PATH}")
        return 0

    return len(event_dirs)


if __name__ == "__main__":
    print(f"Data configuration for {SELECTED_MODEL}")
    print(f"Config file: {_CONFIG_PATH}")
    print(f"Data path: {BASE_PATH}")
    try:
        n_events = validate_data_paths()
        print(f"✓ Data validation successful ({n_events} events found)")
    except FileNotFoundError as e:
        print(f"✗ Data validation failed: {e}")
        exit(1)
