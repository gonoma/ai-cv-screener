import os

import psycopg
from pgvector.psycopg import register_vector

from ..providers.embeddings import EmbeddingModel


class Database:
    _DATABASE_URL: str = os.environ.get("DATABASE_URL", "postgresql://cv:cv@localhost:5433/cv")

    _SCHEMA_SQL: str = f"""
    CREATE TABLE IF NOT EXISTS candidates (
        id               SERIAL PRIMARY KEY,
        source_file      TEXT NOT NULL UNIQUE,
        name             TEXT NOT NULL,
        -- `current_role` cannot be used unquoted: it is a reserved word in
        -- standard SQL (CURRENT_ROLE), and CREATE TABLE fails to parse.
        role             TEXT,
        current_company  TEXT,
        years_experience INTEGER,
        skills           TEXT[] NOT NULL DEFAULT '{{}}',
        institutions     TEXT[] NOT NULL DEFAULT '{{}}'
    );

    CREATE TABLE IF NOT EXISTS chunks (
        id           SERIAL PRIMARY KEY,
        candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
        source_file  TEXT NOT NULL,
        name         TEXT NOT NULL,
        section      TEXT,
        content      TEXT NOT NULL,
        embedding    VECTOR({EmbeddingModel.DIMENSIONS})
    );

    -- Names are not identifiers: the corpus deliberately repeats one, so the
    -- profile route resolves a name to ids and filters on candidate_id.
    CREATE INDEX IF NOT EXISTS chunks_candidate_idx ON chunks (candidate_id);
    CREATE INDEX IF NOT EXISTS candidates_name_idx ON candidates (lower(name));
    CREATE INDEX IF NOT EXISTS candidates_skills_idx ON candidates USING gin (skills);
    """

    def open_connection(self) -> psycopg.Connection:
        connection = psycopg.connect(self._DATABASE_URL, autocommit=True)
        register_vector(connection)
        return connection

    def drop_and_recreate_tables(self, connection: psycopg.Connection) -> None:
        connection.execute("DROP TABLE IF EXISTS chunks")
        connection.execute("DROP TABLE IF EXISTS candidates")
        self._create_tables(connection)

    def _create_tables(self, connection: psycopg.Connection) -> None:
        connection.execute(self._SCHEMA_SQL)

    def count_ingested_rows(self, connection: psycopg.Connection) -> dict[str, int]:
        try:
            candidates, chunks = connection.execute(
                "SELECT (SELECT count(*) FROM candidates), (SELECT count(*) FROM chunks)"
            ).fetchone()
        except psycopg.errors.UndefinedTable:
            return {"candidates": 0, "chunks": 0}
        return {"candidates": candidates, "chunks": chunks}
