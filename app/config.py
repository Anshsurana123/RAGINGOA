"""
App configuration module.
Imports and re-exports configuration from root config.py,
and defines benchmark latency constraints.
"""

import os
import sys
from pathlib import Path

# Ensure root workspace directory is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Re-export all settings from root config
from config import *

# Latency budget in milliseconds for retrieval benchmark (embed + FAISS search)
LATENCY_BUDGET_MS = float(os.getenv("LATENCY_BUDGET_MS", "50.0"))
