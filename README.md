# ai-cv-screener
Welcome to the AI-Powered CV Screener!

Essentially A CV screening tool: a user asks questions in a chat interface and gets answers
grounded in a corpus of CVs, with the source CVs cited.

The project has three parts: a one-shot pipeline that generates a synthetic CV corpus; a backend
that ingests those CVs and answers questions about them; and a web chat
interface.

## Setup

Requires Python 3.12 or newer.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

```
make test    # pytest
make lint    # ruff check
make format  # ruff format
```

Create a .env file and copy-paste the content inside the template `.env.example`.
Fill the API keys and any other required environment variables.

# Design Decisions

## Corpus Generation Pipeline

### The corpus is synthetic on purpose

I can't put real CVs in a repo, so I generate fake ones. Nice side effect: I know
exactly what's in every CV, so later I can check whether the search actually finds
the right people. In other words: retrieval can later be graded against known
answers rather than hand-written guesses.
The evaluation quality of this project rests on that, which is why generation is a 
real pipeline rather than a fixture file.


### Candidate coordinates are fixed in code, before any model call

Before calling the model I pick each person's details in code (job, seniority,
city, where they studied). If you just ask a model for 30 people you get 30
versions of the same person, and then there's nothing to search for. This way I'm
asking it to write a CV for someone I already made up. I deal the options out like
cards so I don't end up with five people in Barcelona and none anywhere else, and
I work out years of experience from seniority so I don't get directors with two
years. It's all seeded, so I get the same 30 people every run.
As a result, a given corpus size always yields the same result. A failing test is 
then a real regression rather than a reshuffle.

### CV Content Builder

The builder returns JSON, not CV text. This way the record becomes an answer key,
it gets rendered into PDF later easily, and it can be checked field by field.
This is quite useful for evals and testing.
If we returned prose or CV text instead we would loose these benefits.

### Photo Builder

Photo Generation has three options, and it tries them in order: Gemini, then Pollinations
which needs no key, then it just draws a coloured circle with the person's
initials (as a local fallback not based on network).
This way we avoid missing a photo on a CV, even with just the initials is better than no
photo at all, also it's not worth killing a run that's already spent quota because of a broken network.

The photos never reach the search. They only exist so the PDFs look like real
documents.
