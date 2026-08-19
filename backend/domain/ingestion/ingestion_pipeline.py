import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from ... import REPO_ROOT
from ...data.database import Database
from ...data.models import CvChunk
from ...providers.embeddings import EmbeddingModel
from .. import candidate_facts
from .candidate_parser import CandidateParser, MisalignedBatch
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
    # tokens it already bought.
    _EXTRACTION_DIRECTORY: Path = REPO_ROOT / "data" / "extractions"

    # CVs per extraction call.
    # The instruction block is the only part of the prompt a batch shares — the CV
    # bodies are ~85% of the tokens and are sent either way, so that 15% is the whole
    # ceiling on what batching can save. At 10 the block is split 10 ways, capturing
    # ~90% of it (13.5% of total); doubling to 20 adds under a percent while widening
    # the window for a misaligned batch, in other words, past 10 you barely save anything
    # and just make the AI more likely to mix people up.
    _BATCH_SIZE: int = 10

    # How much of a batch may be rescued one CV at a time before the batch is
    # declared bad instead.
    # If only a couple of the 10 CVs come back wrong it's worth re-asking those alone,
    # but past that limit the whole batch is treated as junk, a RuntimeError is raised
    # aborting the whole ingest and asks to lower _BATCH_SIZE — nothing from that batch
    # gets cached or written.
    _MAX_SALVAGE_FRACTION: float = 0.3

    # Batches in flight.
    # Concurrency here is tuned for latency, not cost: token spend is identical whether
    # requests go out serially or in parallel, so the limit is the provider's
    # rate/connection tolerance.
    # Raise it if the provider tolerates it and you need more latency (+80 CVs);
    # the work is pure I/O and the local half (embedding) is CPU and runs after.
    _CONCURRENT_READS: int = 8

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

        # Read the whole corpus before dropping anything.
        read_cvs = self._read_corpus(pdf_paths)

        self._database.drop_and_recreate_tables(self.connection)
        ingested = [self._store_cv(read_cv) for read_cv in read_cvs]
        return {
            "candidates": len(ingested),
            "chunks": sum(ingested_file["chunks"] for ingested_file in ingested),
            "extracted": sum(1 for read_cv in read_cvs if not read_cv.from_cache),
            "reused": sum(1 for read_cv in read_cvs if read_cv.from_cache),
            "characters_sent": sum(
                len(read_cv.cv_text) for read_cv in read_cvs if not read_cv.from_cache
            ),
            "files": ingested,
        }

    def _read_corpus(self, pdf_paths: list[Path]) -> list[ReadCv]:
        """The half of ingestion that touches no database, so the half that can run in threads.

        `ingest_corpus` holds a single psycopg connection, which is not safe to
        share across threads, so the split is not a stylistic one: everything
        here is pure I/O against the filesystem and the LLM.

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
            if (
                entry := self._cached_extraction(
                    pdf_path=pdf_path,
                    cv_text=texts[pdf_path],
                )
            )
            is not None
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
        """Extract one batch, keep every record that checks out, re-ask for the rest.

        Cached per CV rather than per batch so a later run reuses whatever
        succeeded, regardless of how the CVs were grouped this time — the batch
        is a transport detail, not a unit of work worth remembering.
        """
        try:
            records = self.candidate_parser.parse_candidates([texts[path] for path in pdf_paths])
        except MisalignedBatch:
            # A batch that answers for one CV out of ten has said nothing about
            # the other nine, and no salvage rule can guess which nine. Asking
            # singly costs a call per CV, which is the expensive path — and
            # still cheaper than failing an ingest that has already paid for
            # every other batch.
            return {
                pdf_path: self._retry_pdf_extraction(
                    pdf_path=pdf_path,
                    cv_text=texts[pdf_path],
                )
                for pdf_path in pdf_paths
            }

        aligned: dict[Path, dict] = {}
        misaligned: list[Path] = []
        for pdf_path, record in zip(pdf_paths, records, strict=True):
            if self._describes_this_cv(
                cv_text=texts[pdf_path],
                record=record,
            ):
                aligned[pdf_path] = record
            else:
                misaligned.append(pdf_path)

        # Always at least one, so a single stray record is still rescued no
        # matter how small the batch — the cap is there to stop a wholesale
        # re-buy, not to make salvage impossible when batches are short.
        salvageable = max(1, int(len(pdf_paths) * self._MAX_SALVAGE_FRACTION))
        if len(misaligned) > salvageable:
            raise RuntimeError(
                f"{len(misaligned)} of {len(pdf_paths)} records in the batch do not match the CV "
                f"they were returned for ({', '.join(p.name for p in misaligned[:3])}...) — "
                "re-asking each one would cost more than the batch. Lower _BATCH_SIZE."
            )

        for pdf_path, record in aligned.items():
            self._cache_extraction(
                pdf_path=pdf_path,
                cv_text=texts[pdf_path],
                candidate_info=record,
            )
        for pdf_path in misaligned:
            aligned[pdf_path] = self._retry_pdf_extraction(
                pdf_path=pdf_path,
                cv_text=texts[pdf_path],
            )
        return aligned

    def _retry_pdf_extraction(self, pdf_path: Path, cv_text: str) -> dict:
        """Re-ask for a single CV, and treat a record-to-cv description failure as a real one."""
        record = self.candidate_parser.parse_candidates([cv_text])[0]
        if not self._describes_this_cv(
            cv_text=cv_text,
            record=record,
        ):
            raise RuntimeError(
                f"{pdf_path.name}: extracted name {record.get('name')!r} does not appear in the "
                "CV, on its own in the prompt — this is not batch misalignment"
            )
        self._cache_extraction(
            pdf_path=pdf_path,
            cv_text=cv_text,
            candidate_info=record,
        )
        return record

    @staticmethod
    def _describes_this_cv(cv_text: str, record: dict) -> bool:
        """Check the record actually describes this CV, which batching can break.

        One call holding ten unrelated people can return them shifted by one, or
        blend two together. Nothing downstream would notice: the row is
        well-formed, the schema validates, and the answer key would simply grade
        the system as wrong at retrieval time. The surname is the cheapest
        anchor that is always present in the source text, and checking it costs
        no tokens at all.

        "Present in the source text" is not enough on its own, though. Models
        reading a CV whose name sits under a heading have returned the *heading*
        as the name — and "PROFILE" passes an appears-in-the-text check
        trivially, because it is right there in the text. Such a row poisons more
        than its own record: the profile route resolves names, and the query
        router treats every name in the corpus as routing vocabulary, so a
        candidate called PROFILE quietly becomes a thing users can ask about.
        Hence two more conditions — a real name is not a section heading, and it
        has more than one part.
        """
        name = str(record.get("name") or "").strip()
        name_parts = name.split()
        if len(name_parts) < 2:
            return False
        if name.lower() in CvTextChunker.SECTION_HEADINGS:
            return False
        return name_parts[-1].lower() in cv_text.lower()

    def _cached_extraction(self, pdf_path: Path, cv_text: str) -> dict | None:
        """
        Return a previous extraction for this CV, or None if there isn't a valid one.
        Validity is the hash of the extracted text, not the filename.
        """
        cache_file = self._extraction_path(pdf_path)
        if not cache_file.exists():
            return None

        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("text_sha256") != self._text_fingerprint(cv_text):
            return None
        # An entry extracted under an older prompt or schema is the right answer
        # to a question no longer being asked — it is missing whatever field was
        # added since, and nothing at read time would reveal that.
        if cached.get("extraction_sha256") != self._extraction_fingerprint():
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
            "extraction_sha256": self._extraction_fingerprint(),
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

    @staticmethod
    def _extraction_fingerprint() -> str:
        """What was asked of the model, so that changing the ask misses the cache."""
        asked = CandidateParser.PROMPT_TEMPLATE + json.dumps(
            CandidateParser.RESPONSE_SCHEMA, sort_keys=True
        )
        return hashlib.sha256(asked.encode("utf-8")).hexdigest()

    def _store_cv(self, read_cv: ReadCv) -> dict:
        candidate_id = self._insert_candidate_row(
            source_file=read_cv.source_file,
            candidate_info=read_cv.candidate_info,
        )
        chunks = self.cv_text_chunker.chunk_cv_text(read_cv.cv_text)
        self._insert_chunk_rows(
            candidate_id=candidate_id,
            source_file=read_cv.source_file,
            candidate_name=read_cv.candidate_info["name"],
            chunks=chunks,
        )

        return {
            "source_file": read_cv.source_file,
            "name": read_cv.candidate_info["name"],
            "chunks": len(chunks),
        }

    def _insert_candidate_row(self, source_file: str, candidate_info: dict) -> int:
        """Store what the model read, plus the arithmetic it was not asked to do.

        Deriving here rather than in the parser keeps the sums out of the
        extraction cache: changing how a career is counted then costs a re-ingest
        of local JSON, not a re-buy of every CV.
        """
        positions = candidate_info.get("positions") or []
        longest = candidate_facts.longest_tenure(positions)
        current = candidate_facts.current_position(positions)
        # A role the extraction left blank is still printed on the CV, at the top
        # of the job it names — so fall back to that rather than store a
        # candidate with no title.
        company = candidate_info.get("current_company") or (current or {}).get("company")
        role = candidate_facts.role_without_company(
            role=candidate_info.get("current_role") or (current or {}).get("role"),
            company=company,
        )

        return self.connection.execute(
            "INSERT INTO candidates (source_file, name, role, current_company, "
            "years_experience, longest_tenure_years, positions, skills, institutions) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            [
                source_file,
                candidate_info["name"],
                role,
                company,
                candidate_facts.career_years(positions),
                candidate_facts.tenure_years(longest) if longest else None,
                Jsonb(positions),
                candidate_info.get("skills") or [],
                candidate_info.get("institutions") or [],
            ],
        ).fetchone()[0]

    def _insert_chunk_rows(
        self,
        candidate_id: int,
        source_file: str,
        candidate_name: str,
        chunks: list[CvChunk],
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
