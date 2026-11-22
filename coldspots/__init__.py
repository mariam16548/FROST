"""FROST: Find Regions Of Substitution Tolerance."""
from .config import Config, load_config
from .pipeline import detect, load_data, run

__version__ = "1.0.0"
__all__ = ["Config", "load_config", "run", "load_data", "detect"]
