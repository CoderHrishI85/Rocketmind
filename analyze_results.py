"""Root entrypoint for RocketMind results analysis and visualization.
"""

import sys
from pathlib import Path

# Add project root and src to path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from scripts.analyze_results import main

if __name__ == "__main__":
    main()

