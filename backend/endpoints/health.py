from fastapi import APIRouter

from ..data.database import Database

router = APIRouter()
database = Database()


@router.get("/health")
def health_check() -> dict:
    with database.open_connection() as connection:
        return {"status": "ok", **database.count_ingested_rows(connection)}
