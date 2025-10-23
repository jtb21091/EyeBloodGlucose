"""Deprecated shim: retrain_from_cleaned.py

This script is no longer used. Please run training.py directly.

Examples:
  python training.py --preclean strong --overwrite-labels --cleanup-clean-files --fast --no-show
  python training.py --preclean strong  # full run without --fast
"""

import sys

msg = (
    "retrain_from_cleaned.py is deprecated. Use training.py with --preclean instead.\n"
    "Example: python training.py --preclean strong --overwrite-labels --cleanup-clean-files --fast --no-show\n"
)
sys.stderr.write(msg)
raise SystemExit(1)
