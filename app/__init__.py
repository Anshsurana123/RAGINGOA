"""
Hacker House Goa 2026 Indic Voice RAG Application Package.
"""

from pathlib import Path
import sys

# Ensure root workspace directory is in sys.path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
