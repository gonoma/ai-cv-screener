import argparse

from dotenv import load_dotenv

from .corpus_generation_pipeline import REPO_ROOT, CorpusGenerationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic CV corpus.")
    parser.add_argument("--count", "-n", type=int, default=30, help="number of CVs (default 30)")
    parser.add_argument(
        "--force", action="store_true", help="regenerate records and photos, ignoring the cache"
    )
    arguments = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    CorpusGenerationPipeline(count=arguments.count, force=arguments.force).run()


if __name__ == "__main__":
    main()
