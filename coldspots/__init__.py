"""Cold-spot detection in disordered protein regions via a two-state HMM."""
from .config import Config, load_config
from .pipeline import run, load_data, detect

__version__ = "1.0.0"
__all__ = ["Config", "load_config", "run", "load_data", "detect"]
