import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import psycopg

from ... import REPO_ROOT
from ...data.database import Database
from ...data.models import CvChunk
from ...providers.embeddings import EmbeddingModel
from .candidate_parser import CandidateParser
from .cv_text_chunker import CvTextChunker
from .pdf_text_extractor import PdfTextExtractor


@dataclass(frozen=True)
class ReadCv:
    source_file: str
    cv_text: str
    candidate_info: dict
    from_cache: bool


class IngestionPipeline:
    _CV_DIRECTORY: Path = REPO_ROOT / "data" / "cvs"

    # One extraction per CV, cached so an interrupted run does not re-spend the
    # calls it already made. Free tiers are metered in tens of requests per day,
    # and the read phase is all-or-nothing, so without this a run that dies on
    # the twenty-sixth CV throws away twenty-five successful calls and cannot be
    # retried until the quota resets.
    _EXTRACTION_DIRECTORY: Path = REPO_ROOT / "data" / "extractions"

    # CVs per extraction call. Free tiers meter calls, not tokens, so a thirty-CV
    # ingest is six requests at this size rather than thirty. Five CVs is roughly
    # 2.5k tokens in and 1.3k out, well inside both providers' limits.
    _BATCH_SIZE: int = 5

    # Batches in flight. The calls are independent, so wall time is set by how
    # many run at once rather than by any local work — embedding is on-CPU and
    # takes seconds. The ceiling is the tightest free tier we target, not the
    # fastest: Gemini allows 5 requests per minute, and going over fails the run
    # on the first wave rather than queueing behind it.
    _CONCURRENT_READS: int = 4

    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection
        self.pdf_text_extractor = PdfTextExtractor()
        self.cv_text_chunker = CvTextChunker()
        self.candidate_parser = CandidateParser()
        self._database = Database()
        self._embedding_model = EmbeddingModel()

    def ingest_corpus(self) -> dict:
        pdf_paths = sorted(self._CV_DIRECTORY.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"no PDFs in {self._CV_DIRECTORY} — run the generator first")

        # Read the whole corpus before dropping anything. Every plausible
        # failure lives in this phase — LLM calls against an endpoint that
        # rate-limits — and dropping first means any one of them failing leaves
        # no corpus at all instead of the previous one.
        read_cvs = self.read_corpus(pdf_paths)

        self._database.drop_and_recreate_tables(self.connection)
        ingested = [self.store_cv(read_cv) for read_cv in read_cvs]
        return {
            "candidates": len(ingested),
            "chunks": sum(ingested_file["chunks"] for ingested_file in ingested),
            # How many CVs cost an LLM call this run. On a resumed run this is
            # the only number that matters, since it is what the quota paid for.
            "extracted": sum(1 for read_cv in read_cvs if not read_cv.from_cache),
            "reused": sum(1 for read_cv in read_cvs if read_cv.from_cache),
            "files": ingested,
        }

    def read_corpus(self, pdf_paths: list[Path]) -> list[ReadCv]:
        """The half of ingestion that touches no database, so the half that can run in threads.

        `ingest_corpus` holds a single psycopg connection, which is not safe to
        share across threads, so the split is not a stylistic one: everything
        here is pure I/O against the filesystem and the LLM, and every write
        happens back on the calling thread in `store_cv`.

        PDF text is pulled for the whole corpus first because it is local and
        costs under a second; only the CVs with no valid cache entry reach the
        model, and those go in batches.
        """
        texts = {
            pdf_path: self.pdf_text_extractor.extract_text_from_pdf(pdf_path)
            for pdf_path in pdf_paths
        }
        cached = {
            pdf_path: entry
            for pdf_path in pdf_paths
            if (entry := self._cached_extraction(pdf_path, texts[pdf_path])) is not None
        }

        uncached = [pdf_path for pdf_path in pdf_paths if pdf_path not in cached]
        batches = [
            uncached[start : start + self._BATCH_SIZE]
            for start in range(0, len(uncached), self._BATCH_SIZE)
        ]
        extracted: dict[Path, dict] = {}
        with ThreadPoolExecutor(max_workers=self._CONCURRENT_READS) as pool:
            for filled in pool.map(partial(self._extract_batch, texts=texts), batches):
                extracted.update(filled)

        return [
            ReadCv(
                source_file=pdf_path.name,
                cv_text=texts[pdf_path],
                candidate_info=cached.get(pdf_path) or extracted[pdf_path],
                from_cache=pdf_path in cached,
            )
            for pdf_path in pdf_paths
        ]

    def _extract_batch(self, pdf_paths: list[Path], texts: dict[Path, str]) -> dict[Path, dict]:
        """Extract one batch, then cache each record separately.

        Cached per CV rather than per batch so a later run reuses whatever
        succeeded, regardless of how the CVs were grouped this time — the batch
        is a transport detail, not a unit of work worth remembering.
        """
        records = self.candidate_parser.parse_candidates([texts[path] for path in pdf_paths])

        extracted = {}
        for pdf_path, record in zip(pdf_paths, records, strict=True):
            self._reject_if_misaligned(pdf_path, texts[pdf_path], record)
            self._cache_extraction(pdf_path, texts[pdf_path], record)
            extracted[pdf_path] = record
        return extracted

    @staticmethod
    def _reject_if_misaligned(pdf_path: Path, cv_text: str, record: dict) -> None:
        """Check the record actually describes this CV, which batching can break.

        One call holding five unrelated people can return them shifted by one,
        or blend two together. Nothing downstream would notice: the row is
        well-formed, the schema validates, and the answer key would simply grade
        the system as wrong at retrieval time. The surname is the cheapest
        anchor that is always present in the source text.
        """
        name_parts = record["name"].split()
        if not name_parts:
            raise RuntimeError(f"{pdf_path.name}: extracted record has no name")

        surname = name_parts[-1]
        if surname.lower() not in cv_text.lower():
            raise RuntimeError(
                f"{pdf_path.name}: extracted name {record['name']!r} does not appear in the CV — "
                "the batch came back misaligned"
            )

    def _cached_extraction(self, pdf_path: Path, cv_text: str) -> dict | None:
        """Return a previous extraction for this CV, or None if there isn't a valid one.

        Validity is the hash of the extracted text, not the filename. A
        regenerated corpus reuses filenames for different people, so keying on
        the name alone would hand new candidates the previous ones' skills —
        silently, and in a way no test would catch. Hashing the text means a
        changed CV simply misses, which is why there is no --force flag to
        forget.
        """
        cache_file = self._extraction_path(pdf_path)
        if not cache_file.exists():
            return None

        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("text_sha256") != self._text_fingerprint(cv_text):
            return None
        return cached["candidate_info"]

    def _cache_extraction(self, pdf_path: Path, cv_text: str, candidate_info: dict) -> None:
        """Write via a temporary file, because this exists to survive interruption.

        A run killed mid-write would otherwise leave truncated JSON that fails
        the *next* run on read — turning the cache into a new way to lose the
        corpus. os.replace is atomic on POSIX.
        """
        self._EXTRACTION_DIRECTORY.mkdir(parents=True, exist_ok=True)
        cache_file = self._extraction_path(pdf_path)
        payload = {
            "source_file": pdf_path.name,
            "text_sha256": self._text_fingerprint(cv_text),
            "candidate_info": candidate_info,
        }
        partial_file = cache_file.parent / f"{cache_file.stem}.{os.getpid()}.partial"
        partial_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(partial_file, cache_file)

    def _extraction_path(self, pdf_path: Path) -> Path:
        return self._EXTRACTION_DIRECTORY / f"{pdf_path.stem}.json"

    @staticmethod
    def _text_fingerprint(cv_text: str) -> str:
        return hashlib.sha256(cv_text.encode("utf-8")).hexdigest()

    def store_cv(self, read_cv: ReadCv) -> dict:
        candidate_id = self._insert_candidate_row(read_cv.source_file, read_cv.candidate_info)

        chunks = self.cv_text_chunker.chunk_cv_text(read_cv.cv_text)
        self._insert_chunk_rows(
            candidate_id, read_cv.source_file, read_cv.candidate_info["name"], chunks
        )
        return {
            "source_file": read_cv.source_file,
            "name": read_cv.candidate_info["name"],
            "chunks": len(chunks),
        }

    def _insert_candidate_row(self, source_file: str, candidate_info: dict) -> int:
        return self.connection.execute(
            "INSERT INTO candidates (source_file, name, role, current_company, "
            "years_experience, skills, institutions) VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id",
            [
                source_file,
                candidate_info["name"],
                candidate_info.get("current_role"),
                candidate_info.get("current_company"),
                candidate_info.get("years_experience"),
                candidate_info.get("skills") or [],
                candidate_info.get("institutions") or [],
            ],
        ).fetchone()[0]

    def _insert_chunk_rows(
        self, candidate_id: int, source_file: str, candidate_name: str, chunks: list[CvChunk]
    ) -> None:
        vectors = self._embedding_model.embed_texts([chunk.content for chunk in chunks])
        with self.connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunks (candidate_id, source_file, name, section, content, "
                "embedding) VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (
                        candidate_id,
                        source_file,
                        candidate_name,
                        chunk.section,
                        chunk.content,
                        str(vector),
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
