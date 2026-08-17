# ai-cv-screener
Welcome to the AI-Powered CV Screener!

Essentially A CV screening tool: a user asks questions in a chat interface and gets answers
grounded in a corpus of CVs, with the source CVs cited.

The project has three parts: a one-shot pipeline that generates a synthetic CV corpus; a backend
that ingests those CVs and answers questions about them; and a web chat
interface.

## Setup

Requires Python 3.12 or newer. WeasyPrint needs a couple of system libraries:

```
brew install pango libffi            # macOS
sudo apt install libpango-1.0-0 libpangoft2-1.0-0    # Debian/Ubuntu
```

Create a Virtual Environment with installed dependencies, activate it,
and create a .env file that needs to be filled with your personal keys.
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # add your GEMINI_API_KEY
```

Makefile commands for simplicity and ease of use:

```
make test                # to check a corpus is actually varied
make lint                # ruff check
make format              # ruff format
make generate            # generates 30 CVs
make generate COUNT=5    # fewer, while testing changes
```

You can also run the generation command from the terminal:
```
python -m data_generation.run --count 30
```

With the `--force` flag it skips everything already on disk:

```
python -m data_generation.run --count 30 --force
```

To generate CVs you need:

- `GEMINI_API_KEY` from Google AI Studio, and `TEXT_PROVIDER=gemini`
- `OPENROUTER_API_KEY` from OpenRouter, and `TEXT_PROVIDER=openrouter`

Photos don't need a key at all. `IMAGE_PROVIDER` picks where to start, and
anything that fails falls through to the next option.

The tests don't rely on external network APIs.

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

### CV Renderer

There are five different CV layouts, and that's not for looks. Getting text back
out of a PDF is the step that quietly goes wrong, and it goes wrong differently
depending on how the page is arranged. Thirty identical CVs would test one code
path thirty times, hence why we have 5 different templates that are quite 
different in structure. The templates cycle by position rather than at random, 
so all five always show up.

### Corpus Generation Pipeline

It writes:

- `data/cvs/` — one PDF per candidate
- `data/photos/` — one JPEG per candidate
- `data/ground_truth.json` — the answer key, i.e. exactly what went into each CV

Two candidates sharing a name is something a real pile of CVs contains anyway, 
and the record still differs everywhere else, so there's nothing here re-requesting 
a record until it looks distinct enough.
Same for photos: each candidate has their own appearance line, and the headshot
that comes back is taken as-is.

The answer key gets rewritten after every single CV rather than once at the end.
If a run dies on number 18, I keep the 17 I already paid for.
