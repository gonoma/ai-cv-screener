import pytest

from backend.domain.ingestion import IngestionPipeline

INFO = {"name": "Ada Lovelace", "skills": ["Python"], "institutions": ["UPC"]}


class CountingParser:
    """Stands in for the LLM, recording calls and CVs-per-call separately.

    Both numbers matter: batching is supposed to cut the number of *calls*
    without changing how many CVs get extracted.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.cvs_seen = 0

    def parse_candidates(self, cv_texts: list[str]) -> list[dict]:
        # Reads the name back out of each CV rather than numbering by position:
        # a stub that ignores its input would itself look misaligned to the
        # check in the pipeline, and would hide a real ordering bug.
        self.calls += 1
        self.cvs_seen += len(cv_texts)
        return [dict(INFO, name=text.split(" —")[0]) for text in cv_texts]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """A pipeline with the slow halves replaced: no PDF, no LLM, no database.

    __new__ rather than __init__ because the constructor opens a database
    connection and loads the embedding model, neither of which this touches.
    """
    monkeypatch.setattr(IngestionPipeline, "_EXTRACTION_DIRECTORY", tmp_path / "extractions")
    built = IngestionPipeline.__new__(IngestionPipeline)
    built.candidate_parser = CountingParser()
    built.pdf_text_extractor = type("Stub", (), {"extract_text_from_pdf": staticmethod(read_text)})
    return built


def read_text(pdf_path):
    return pdf_path.read_text(encoding="utf-8")


def write_cvs(tmp_path, count: int) -> list:
    paths = []
    for index in range(count):
        cv = tmp_path / f"ada-lovelace-{index}.pdf"
        cv.write_text(f"Ada Lovelace {index} — Python, UPC", encoding="utf-8")
        paths.append(cv)
    return paths


def test_a_batch_of_cvs_costs_one_call(pipeline, tmp_path):
    paths = write_cvs(tmp_path, IngestionPipeline._BATCH_SIZE)

    read = pipeline.read_corpus(paths)

    assert pipeline.candidate_parser.calls == 1
    assert pipeline.candidate_parser.cvs_seen == len(paths)
    assert [cv.source_file for cv in read] == [path.name for path in paths]


def test_second_run_costs_no_llm_call(pipeline, tmp_path):
    paths = write_cvs(tmp_path, 3)

    first = pipeline.read_corpus(paths)
    second = pipeline.read_corpus(paths)

    assert pipeline.candidate_parser.calls == 1
    assert [cv.from_cache for cv in first] == [False, False, False]
    assert [cv.from_cache for cv in second] == [True, True, True]
    assert [cv.candidate_info for cv in second] == [cv.candidate_info for cv in first]


def test_only_the_uncached_cvs_are_sent(pipeline, tmp_path):
    paths = write_cvs(tmp_path, 3)
    pipeline.read_corpus(paths[:2])

    pipeline.read_corpus(paths)

    assert pipeline.candidate_parser.calls == 2
    assert pipeline.candidate_parser.cvs_seen == 3, "a cached CV was re-sent to the model"


def test_changed_cv_text_is_not_served_from_cache(pipeline, tmp_path):
    """The failure this cache could plausibly cause, rather than the happy path.

    Regenerating the corpus reuses filenames for different people, so a cache
    keyed on the name alone would hand a new candidate the previous one's
    skills — silently, and gradeable as correct by every eval.
    """
    paths = write_cvs(tmp_path, 1)
    pipeline.read_corpus(paths)

    paths[0].write_text("Someone Else — Rust, MIT", encoding="utf-8")
    reread = pipeline.read_corpus(paths)

    assert pipeline.candidate_parser.calls == 2
    assert reread[0].from_cache is False


def test_a_record_that_does_not_match_its_cv_is_rejected(pipeline, tmp_path):
    """Misalignment is the risk batching adds, and it is otherwise silent.

    A batch returned shifted by one produces well-formed rows that validate
    against the schema; only the answer key would ever disagree, and by then it
    reads as a retrieval failure rather than an ingestion one.
    """
    paths = write_cvs(tmp_path, 1)
    pipeline.candidate_parser.parse_candidates = lambda texts: [dict(INFO, name="Grace Hopper")]

    with pytest.raises(RuntimeError, match="misaligned"):
        pipeline.read_corpus(paths)
