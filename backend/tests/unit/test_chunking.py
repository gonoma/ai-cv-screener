from backend.domain.ingestion import CvTextChunker

CHUNKER = CvTextChunker()


def test_empty_input_produces_no_chunks() -> None:
    assert CHUNKER.chunk_cv_text("") == []
    assert CHUNKER.chunk_cv_text("   \n  ") == []


def test_headings_become_sections() -> None:
    text = "Ada Lovelace\nada@example.com\nEXPERIENCE\nEngineer at Acme\nEDUCATION\nBSc, UPC"
    sections = CHUNKER.split_into_sections(text)
    assert [heading for heading, _ in sections] == [None, "Experience", "Education"]
    assert "ada@example.com" in sections[0][1]
    assert "UPC" in sections[2][1]


def test_text_without_headings_is_one_section() -> None:
    assert [h for h, _ in CHUNKER.split_into_sections("just a wall of text")] == [None]


def test_section_longer_than_the_window_is_split() -> None:
    body = "x" * (CvTextChunker.CHUNK_CHARS * 2)
    windows = CHUNKER.split_into_overlapping_windows(body)
    assert len(windows) > 1
    assert all(len(w) <= CvTextChunker.CHUNK_CHARS for w in windows)


def test_consecutive_windows_actually_overlap() -> None:
    # Asserts on shared content, not on arithmetic derived from CHUNK_OVERLAP:
    # that would confirm the code agrees with itself and pass at overlap 0.
    assert CvTextChunker.CHUNK_OVERLAP > 0, "no overlap means boundary facts are lost"

    body = "".join(f"{i:05d}." for i in range(400))
    windows = CHUNKER.split_into_overlapping_windows(body)
    assert len(windows) > 1

    overlap = CvTextChunker.CHUNK_OVERLAP
    assert windows[0][-overlap:] == windows[1][:overlap], "windows share no text"


def test_short_body_is_not_split() -> None:
    assert CHUNKER.split_into_overlapping_windows("short") == ["short"]


def test_chunks_carry_their_section() -> None:
    text = "SKILLS\nPython, Kubernetes\nEDUCATION\nBSc"
    assert [c.section for c in CHUNKER.chunk_cv_text(text)] == ["Skills", "Education"]
