from fastapi import APIRouter

from ..data.database import Database
from ..domain.ingestion import IngestionPipeline

router = APIRouter()
database = Database()


@router.post("/ingest")
def ingest_corpus() -> dict:
    with database.open_connection() as connection:
        return IngestionPipeline(connection).ingest_corpus()
