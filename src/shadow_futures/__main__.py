"""
Module entry point for shadow_futures package.

Allows running via: python -m shadow_futures <command>
"""

import sys
from shadow_futures.cli import main

if __name__ == "__main__":
    sys.exit(main())

