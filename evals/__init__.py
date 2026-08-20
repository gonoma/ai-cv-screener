"""The evaluation suite: does the system read the CVs right, and answer right?

Two entry points, each explained at the top of its own file:

    make eval-extraction   ->  extraction.py   (free; nothing needs to be running)
    make evals             ->  run.py          (asks a live backend real questions)

This file runs first whenever either is imported, and its only job is to load
.env — so that settings like BACKEND_API are in place before any module reads
them. It mirrors what backend/__init__.py does for the backend.
"""

from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
