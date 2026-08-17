from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import REPO_ROOT
from .endpoints import chat, health, ingest

# No CORS middleware: the frontend proxies /api to this process, so requests
# are same-origin. A cross-origin error means the Vite proxy is wrong.
app = FastAPI(title="CV Screening")
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(chat.router)

# The cited PDFs, so a source chip opens the document it names. A citation the
# reader cannot open is a claim about a source rather than a source.
# check_dir=False because a fresh clone has no corpus yet, and the API refusing
# to start would hide the /health check that tells you to generate one.
app.mount(
    "/cvs",
    StaticFiles(directory=REPO_ROOT / "data" / "cvs", check_dir=False),
    name="cvs",
)
