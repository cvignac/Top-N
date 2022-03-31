from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).absolute().parents[0]
# Config directory
CONFIG_DIR = ROOT_DIR.joinpath("config")
# Data directory
DATA_DIR = ROOT_DIR.joinpath("data")
