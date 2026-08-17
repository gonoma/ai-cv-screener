import json
import re
import time
from pathlib import Path
from typing import Any

from .builders import CvContentBuilder, ProfilePhotoBuilder
from .models.candidate_specs import CandidateSpec, build_spec_matrix
from .models.cv_records import CORPUS_YEAR, Candidate
from .rendering import CvPdfRenderer

REPO_ROOT = Path(__file__).resolve().parents[1]


class CorpusGenerationPipeline:
    _CV_DIRECTORY: Path = REPO_ROOT / "data" / "cvs"
    _PHOTO_DIRECTORY: Path = REPO_ROOT / "data" / "photos"
    _GROUND_TRUTH_PATH: Path = REPO_ROOT / "data" / "ground_truth.json"

    _REQUEST_DELAY_SECONDS: float = 4.0
    _RATE_LIMIT_ATTEMPTS: int = 5

    def __init__(self, count: int, force: bool = False) -> None:
        self.count = count
        self.force = force
        self.cv_content_builder = CvContentBuilder()
        self.profile_photo_builder = ProfilePhotoBuilder()
        self.cv_pdf_renderer = CvPdfRenderer()

        self.records: list[dict[str, Any]] = []
        self.used_filenames: set[str] = set()

    def run(self) -> None:
        """
        Build the corpus one CV at a time, reusing whatever is already on disk.

        Resumability is enforced.

        A cached record is never re-requested, an existing photo skips the
        image model, and the answer key is rewritten after each CV rather than
        once at the end.
        """
        self._CV_DIRECTORY.mkdir(parents=True, exist_ok=True)
        self._PHOTO_DIRECTORY.mkdir(parents=True, exist_ok=True)
        if self.force:
            self._delete_generated_files()

        cached_records = self._load_cached_records()
        specs = build_spec_matrix(self.count)

        for index in range(self.count):
            cached = cached_records.get(self._cv_id(index))
            if cached:
                self._materialise_cv(
                    index=index,
                    candidate=Candidate.from_ground_truth_record(cached),
                    source="cached",
                )
            else:
                self._materialise_cv(
                    index=index,
                    candidate=self._generate_candidate(specs[index]),
                    source="generated",
                )

        print(
            f"\n{len(self.records)} PDFs in {self._CV_DIRECTORY}, "
            f"answer key in {self._GROUND_TRUTH_PATH}"
        )

    def _generate_candidate(self, spec: CandidateSpec) -> Candidate:
        for attempt in range(self._RATE_LIMIT_ATTEMPTS):
            try:
                candidate = self.cv_content_builder.build_candidate(spec, self.records)
            except Exception as error:
                wait_seconds = self._retry_after_seconds(error)
                if wait_seconds is None or attempt == self._RATE_LIMIT_ATTEMPTS - 1:
                    raise
                print(f"      rate limited, waiting {wait_seconds:.0f}s")
                time.sleep(wait_seconds)
                continue

            return candidate

        raise AssertionError("unreachable: the final attempt re-raises")

    def _materialise_cv(self, index: int, candidate: Candidate, source: str) -> None:
        cv_id = self._cv_id(index)
        photo_path = self._PHOTO_DIRECTORY / f"{cv_id}.jpg"
        if not photo_path.exists():
            photo_path.write_bytes(self._build_photo(candidate, index))

        filename = self._unique_pdf_filename(candidate.name, cv_id)
        template = self.cv_pdf_renderer.template_for_index(index)
        self.cv_pdf_renderer.render_to_pdf(
            candidate=candidate,
            photo_data_uri=self.profile_photo_builder.photo_as_data_uri(photo_path),
            template=template,
            output_path=self._CV_DIRECTORY / filename,
        )
        self.records.append(candidate.to_ground_truth_record(cv_id, filename, template))

        self._save_ground_truth()
        print(f"  [{index + 1}/{self.count}] {cv_id} {filename} ({template}, {source})")

    def _build_photo(self, candidate: Candidate, index: int) -> bytes:
        """One headshot for this candidate, normalised to a small JPEG.

        The seed is the slot number, so a rerun of the same slot draws the same
        photo and the appearance the model wrote is what varies them.
        """
        prompt = self.profile_photo_builder.build_photo_prompt(
            presentation=candidate.presentation,
            appearance=candidate.appearance,
            index=index,
        )
        raw_image = self.profile_photo_builder.build_photo(
            prompt=prompt,
            initials=self.profile_photo_builder.initials_from_name(candidate.name),
            seed=index,
        )

        return self.profile_photo_builder.normalise_to_small_jpeg(raw_image)

    def _unique_pdf_filename(self, name: str, cv_id: str) -> str:
        """Two candidates can share a name, and the slug would collide with it."""
        filename = self.cv_pdf_renderer.pdf_filename_for(name)
        if filename in self.used_filenames:
            filename = filename.replace(".pdf", f"-{cv_id}.pdf")
        self.used_filenames.add(filename)
        return filename

    def _save_ground_truth(self) -> None:
        self._GROUND_TRUTH_PATH.write_text(
            json.dumps(
                {
                    "corpus_year": CORPUS_YEAR,
                    "count": len(self.records),
                    "candidates": self.records,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        """
        Returns a rate limit suggested waiting time from the provider's error message.
        Returns None for every other failure.
        """
        message = str(error)
        if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
            return None
        suggested_wait = re.search(r"retry in ([\d.]+)s", message)
        return float(suggested_wait.group(1)) + 1 if suggested_wait else 30.0

    @staticmethod
    def _cv_id(index: int) -> str:
        return f"cv-{index + 1:02d}"

    def _load_cached_records(self) -> dict[str, dict[str, Any]]:
        if self.force or not self._GROUND_TRUTH_PATH.exists():
            return {}
        payload = json.loads(self._GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
        return {record["id"]: record for record in payload["candidates"]}

    def _delete_generated_files(self) -> None:
        generated_files = list(self._CV_DIRECTORY.glob("*.pdf")) + list(
            self._PHOTO_DIRECTORY.glob("*.jpg")
        )
        for path in generated_files:
            path.unlink()
        print(f"--force: removed {len(generated_files)} generated file(s)")
