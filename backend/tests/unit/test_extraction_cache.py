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
    paths = write_cvs(
        tmp_path=tmp_path,
        count=IngestionPipeline._BATCH_SIZE,
    )

    read = pipeline._read_corpus(paths)

    assert pipeline.candidate_parser.calls == 1
    assert pipeline.candidate_parser.cvs_seen == len(paths)
    assert [cv.source_file for cv in read] == [path.name for path in paths]


def test_second_run_costs_no_llm_call(pipeline, tmp_path):
    paths = write_cvs(
        tmp_path=tmp_path,
        count=3,
    )

    first = pipeline._read_corpus(paths)
    second = pipeline._read_corpus(paths)

    assert pipeline.candidate_parser.calls == 1
    assert [cv.from_cache for cv in first] == [False, False, False]
    assert [cv.from_cache for cv in second] == [True, True, True]
    assert [cv.candidate_info for cv in second] == [cv.candidate_info for cv in first]


def test_only_the_uncached_cvs_are_sent(pipeline, tmp_path):
    paths = write_cvs(
        tmp_path=tmp_path,
        count=3,
    )
    pipeline._read_corpus(paths[:2])

    pipeline._read_corpus(paths)

    assert pipeline.candidate_parser.calls == 2
    assert pipeline.candidate_parser.cvs_seen == 3, "a cached CV was re-sent to the model"


def test_changed_cv_text_is_not_served_from_cache(pipeline, tmp_path):
    """The failure this cache could plausibly cause, rather than the happy path.

    Regenerating the corpus reuses filenames for different people, so a cache
    keyed on the name alone would hand a new candidate the previous one's
    skills — silently, and gradeable as correct by every eval.
    """
    paths = write_cvs(
        tmp_path=tmp_path,
        count=1,
    )
    pipeline._read_corpus(paths)

    paths[0].write_text("Someone Else — Rust, MIT", encoding="utf-8")
    reread = pipeline._read_corpus(paths)

    assert pipeline.candidate_parser.calls == 2
    assert reread[0].from_cache is False


def test_a_record_that_does_not_match_its_cv_is_rejected(pipeline, tmp_path):
    """Misalignment is the risk batching adds, and it is otherwise silent.

    A batch returned shifted by one produces well-formed rows that validate
    against the schema; only the answer key would ever disagree, and by then it
    reads as a retrieval failure rather than an ingestion one.

    A record that is still wrong when the CV is alone in the prompt is not a
    batching artefact, so it stops the run rather than being retried forever.
    """
    paths = write_cvs(
        tmp_path=tmp_path,
        count=1,
    )
    pipeline.candidate_parser.parse_candidates = lambda texts: [dict(INFO, name="Grace Hopper")]

    with pytest.raises(RuntimeError, match="not batch misalignment"):
        pipeline._read_corpus(paths)


def test_one_bad_record_does_not_cost_the_rest_of_its_batch(pipeline, tmp_path):
    """The expensive failure on a token budget: paying twice for records that were fine.

    Rejecting the whole batch used to discard nine good extractions to punish
    one bad one, and the next run bought all ten again. Now the nine are cached
    and only the tenth is re-asked.
    """
    paths = write_cvs(
        tmp_path=tmp_path,
        count=3,
    )
    good = pipeline.candidate_parser.parse_candidates

    def one_record_comes_back_shifted(texts):
        records = good(texts)
        if len(texts) > 1:  # the batch call, not the single retry
            records[1] = dict(INFO, name="Grace Hopper")
        return records

    pipeline.candidate_parser.parse_candidates = one_record_comes_back_shifted
    read = pipeline._read_corpus(paths)

    assert [cv.candidate_info["name"] for cv in read] == [
        "Ada Lovelace 0",
        "Ada Lovelace 1",
        "Ada Lovelace 2",
    ]
    # One batch, plus a retry for the single CV that came back wrong.
    assert pipeline.candidate_parser.calls == 2
    assert pipeline.candidate_parser.cvs_seen == 4, "a good record was re-sent to the model"


def test_a_wholly_misaligned_batch_is_not_re_bought_one_cv_at_a_time(pipeline, tmp_path):
    """Salvage has to stay cheaper than the batch it rescues, or it is not salvage.

    Re-asking every CV in a bad batch costs a full call each — an order of
    magnitude more than the batch, and that many sequential round trips. It
    reads as a hang and bills like a stampede, so a mostly-wrong batch is
    reported instead.
    """
    paths = write_cvs(
        tmp_path=tmp_path,
        count=10,
    )
    calls = []

    def every_record_is_the_wrong_person(texts):
        calls.append(len(texts))
        return [dict(INFO, name="Grace Hopper") for _ in texts]

    pipeline.candidate_parser.parse_candidates = every_record_is_the_wrong_person

    with pytest.raises(RuntimeError, match="cost more than the batch"):
        pipeline._read_corpus(paths)

    assert calls == [10], "re-asked for CVs after giving up on the batch"


def test_nothing_from_a_rejected_batch_is_left_in_the_cache(pipeline, tmp_path):
    """A rejected batch must not leave a trail the next run trusts."""
    paths = write_cvs(
        tmp_path=tmp_path,
        count=10,
    )
    good = pipeline.candidate_parser.parse_candidates
    pipeline.candidate_parser.parse_candidates = lambda texts: [
        record if index == 0 else dict(INFO, name="Grace Hopper")
        for index, record in enumerate(good(texts))
    ]

    with pytest.raises(RuntimeError, match="cost more than the batch"):
        pipeline._read_corpus(paths)

    assert list(IngestionPipeline._EXTRACTION_DIRECTORY.glob("*.json")) == []


def test_the_salvaged_records_are_cached_too(pipeline, tmp_path):
    """What makes the salvage worth anything: the second run pays for nothing."""
    paths = write_cvs(
        tmp_path=tmp_path,
        count=3,
    )
    good = pipeline.candidate_parser.parse_candidates
    pipeline.candidate_parser.parse_candidates = lambda texts: (
        [dict(INFO, name="Grace Hopper")] + good(texts)[1:] if len(texts) > 1 else good(texts)
    )

    pipeline._read_corpus(paths)
    calls_after_first_run = pipeline.candidate_parser.calls
    second = pipeline._read_corpus(paths)

    assert pipeline.candidate_parser.calls == calls_after_first_run
    assert [cv.from_cache for cv in second] == [True, True, True]


def test_a_section_heading_returned_as_a_name_is_rejected(pipeline, tmp_path):
    """The guard's blind spot: a heading always appears in the CV it was read from.

    "PROFILE" passes an appears-in-the-text check trivially, and the resulting
    row is well-formed. It would reach the corpus as a candidate — and the query
    router treats every name in the corpus as routing vocabulary.
    """
    paths = write_cvs(
        tmp_path=tmp_path,
        count=1,
    )
    paths[0].write_text("PROFILE\nAda Lovelace — Python, UPC", encoding="utf-8")
    pipeline.candidate_parser.parse_candidates = lambda texts: [dict(INFO, name="PROFILE")]

    with pytest.raises(RuntimeError, match="not batch misalignment"):
        pipeline._read_corpus(paths)


def test_a_one_word_name_is_rejected(pipeline, tmp_path):
    paths = write_cvs(
        tmp_path=tmp_path,
        count=1,
    )
    pipeline.candidate_parser.parse_candidates = lambda texts: [dict(INFO, name="Lovelace")]

    with pytest.raises(RuntimeError, match="not batch misalignment"):
        pipeline._read_corpus(paths)
