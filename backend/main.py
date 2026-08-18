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

# This makes the PDF files in data/cvs downloadable from the web app,
# so when the chatbot says "I got this from Priya's CV," you can click
# and actually open it. check_dir=False means: if that folder doesn't exist yet,
# don't crash on startup — just boot anyway so /health can tell you to generate the CVs first.
app.mount(
    "/cvs",
    StaticFiles(directory=REPO_ROOT / "data" / "cvs", check_dir=False),
    name="cvs",
)
