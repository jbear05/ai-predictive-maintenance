"""Source modules for predictive maintenance application."""

import sys
from pathlib import Path

# Add parent directory to path so src modules can import config and loaders
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
