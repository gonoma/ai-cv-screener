from pathlib import Path

from dotenv import load_dotenv

# Loaded here so it happens on any `import backend.*`, before a module reads os.environ.
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
