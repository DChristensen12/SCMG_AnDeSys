"""
Clear StrawberryWatch's local data cache.

Useful when:
  - The cache schema is stale (you added a new feature, old CSV columns mismatch)
  - You want a clean rebuild for debugging
  - Disk space pressure (shouldn't happen with rolling window, but just in case)

Usage:
    python -m scripts.clear_cache
"""

import os
import sys

# Allow running from project root: `python -m scripts.clear_cache`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config


def main():
    path = Config.DATA_FILE
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        os.remove(path)
        print(f"Cleared {path} ({size_kb:.1f} KB)")
    else:
        print(f"No cache file at {path} — nothing to clear.")


if __name__ == "__main__":
    main()
    