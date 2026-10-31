"""The models, the Assistant Axis repository's commit, the layers, and where data is kept."""
import os
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

UPSTREAM_URL = "https://github.com/safety-research/assistant-axis.git"
UPSTREAM_COMMIT = "a98961956072224eaf244eb289d6c01700b63795"

N_LAYERS = 36
# The middle layer: the Assistant Axis code's default for a model not in its MODEL_CONFIGS (total_layers // 2)
TARGET_LAYER = 18


def data_dir() -> Path:
    """The data directory: $PN_DATA_DIR, or ./data. Its layout is in the README."""
    return Path(os.environ.get("PN_DATA_DIR", "data")).resolve()


def upstream_dir() -> Path:
    """Where the Assistant Axis repository is checked out: $PN_UPSTREAM_DIR, or upstream/ in the data directory."""
    return Path(os.environ.get("PN_UPSTREAM_DIR", data_dir() / "upstream" / "assistant-axis")).resolve()
