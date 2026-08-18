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

The backend needs a database. Postgres with the `pgvector` extension runs in Docker,
so Docker Desktop (or any Docker daemon) has to be running before `make db`.

Makefile commands for simplicity and ease of use:

```
make db                  # Postgres + pgvector in Docker, waits until it's ready
make api                 # the backend on http://localhost:8000, with reload
make ingest              # reads every CV in data/cvs into the database
make test                # generation and backend suites
make lint                # ruff check
make format              # ruff format
make generate            # generates 30 CVs
make generate COUNT=5    # fewer, while testing changes
```

The order matters the first time: `make generate` to get a corpus, `make db` to get
somewhere to put it, `make api` in one terminal, then `make ingest` in another.
`GET /health` tells you how many candidates and chunks are currently in the database.

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

The backend uses the same `TEXT_PROVIDER` key, for reading fields out of a CV and for
writing the answers. Embeddings run locally, so search itself costs nothing and needs
no key. `DATABASE_URL` is already in `.env.example` and matches what `make db` starts.

The backend is tuned to spend as few **tokens** as possible, not as few calls — see
[Backend](#backend) below. The knobs that control that are all in `.env.example`:
`LLM_MAX_OUTPUT_TOKENS` caps the answer,
`GEMINI_THINKING_LEVEL` keeps the model from buying a reasoning budget it doesn't need,
`SEMANTIC_SEARCH_TOP_K` is how many chunks an open-ended question retrieves, and
`LLM_CACHE=1` replays identical calls off disk while you're tuning prompts.

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

## Backend

The backend is optimized for tokens, and three rules explain almost everything below: 
* don't send a word twice
* don't send a word nobody asked for
* don't call a model when something cheaper knows the answer.

### Ingesting the CVs

Turning a pile of PDFs into something answerable. Runs once, with `make ingest`.

```
PDF ─▶ text ─▶ [LLM] fields ─▶ chunks ─▶ vectors ─▶ Postgres
```

**1. Get the text out.** The fast reader (`pypdf`) handles almost every page; only
when it comes back nearly empty the slow, careful one (`pdfplumber`) is executed,
page by page, keeping whichever read more words. The two agree to within 1% of word
count, so the fallback fires on a broken page rather than on a hard layout.

**2. Ask a model for the facts.** Name, role, employer, years, skills, schools —
pulled into a fixed JSON shape. Ten CVs ride in one call, not to save calls but so the
instructions are sent once instead of ten times.

When you ask the AI about 10 CVs at once, it can mix up the order and give you Bob's details 
labeled as Anna's — so each result is only trusted if the surname it claims really shows up in that CV. 
The mismatched one gets re-asked by itself, and if it's still wrong, everything stops, because saving 
the wrong person's info would silently poison every future answer.

**3. Cut the text into chunks.** Embedding a whole CV gives one blurry vector that
matches everything and nothing. The cuts follow the CV's own headings, and each piece
repeats the tail of the one before it, so a sentence on a boundary survives whole
somewhere.

**4. Turn the chunks into vectors, locally.** A small model (`bge-small-en-v1.5`) runs
on the machine through ONNX. Hosted embeddings are billed per token like everything
else — on the whole corpus at every ingest, and on every question forever after. This
is the largest token bill in the system, and it simply doesn't exist.
In other words, the piece of a RAG system that usually costs the most here costs nothing at all.

**5. Store it.** Rows and vectors both land in Postgres, and the whole corpus is read
*before* anything is dropped — so a failure halfway leaves you the corpus you had five
minutes ago rather than none at all.

> **30 CVs = ~50k characters → 3 LLM calls, ~13k tokens in, ~4k out.** Once. Records
> are cached under a hash of the CV's text, so a re-ingest is free unless the text
> changed — and a regenerated CV reusing a filename misses the cache instead of
> inheriting the previous person's skills.

### Running it twice, and running it badly

The PDFs are the only thing that really matters. Everything else — the cache, the
tables, the vectors — can be thrown away and rebuilt from them, so there's no state
here worth being precious about.

- **Ingest again, unchanged.** Nearly free. Every CV is already in
  `data/extractions/`, so no model is called; the chunks and vectors are just
  recomputed locally, which costs seconds and no tokens.
- **Delete `data/extractions/` first.** You buy the whole corpus again — 3 calls,
  ~13k tokens, about 30 seconds on Gemini. Nothing breaks; it's just the one thing
  that costs money.
- **It dies halfway.** You keep the corpus you already had. The tables are only
  dropped once every CV has been read, so a failure in the expensive half leaves the
  previous 25 candidates answering questions as if nothing happened.
- **It succeeds.** `candidates` and `chunks` are dropped and rebuilt from empty, never
  patched. Regenerating the corpus invalidates every candidate id anyway, so an
  incremental path would just be a second way for the rows and the vectors to disagree.

### Answering a question

```
question ─▶ route ─▶ fetch context ─▶ [LLM] answer ─▶ stream to the browser
```

**1. Work out what kind of question it is.** Not every question is a search. "Who
knows Python" wants *everyone* who knows Python; "summarise Ana Silva" wants one
person; "who'd suit a startup" is a judgement call. Pick wrong and the answer is
confidently wrong:

| route | for | fetches |
|---|---|---|
| `structured` | filters and counts | a SQL `WHERE`, **every** match |
| `profile` | one named person | that person's CV |
| `semantic` | open-ended, qualitative | the 8 nearest chunks |

**This step is usually free.** Recruiters name things — a person, a technology, a
university — and the database already knows every name, skill and school in the
corpus. Matching against that list is a string comparison, and string comparisons cost
nothing. The model only sees questions the corpus can't explain, which is where it was
earning its keep anyway.

**2. Fetch only what was asked for.** `structured` returns every match, because
similarity search cannot count: ask a vector search "who knows Python" and it returns
8 chunks whether 8 people match or 18, admitting nothing. But it sends the *narrowest*
row that still answers — names and skills, not employers and universities too.

`profile` resolves a name to candidate **ids** first, because names aren't
identifiers: the corpus can have two different people with the same one, and
filtering on the name would blend two careers into one incoherent profile. Their
chunks are stitched back together with the repeated edges removed.

**3. Answer, grounded and cited.** The retrieved text is the only information in
existence: no general knowledge, no filling-in, and "it's not in the CVs" is a valid
answer. 
Each claim carries its source file and the API serves those PDFs. The reply is
length-capped and thinking is minimal, since output costs several times input and
neither job here is a reasoning problem. It streams back as Server-Sent Events, route
label first, so the UI can say what it decided while the words are still arriving.

## What one question costs
  
  **One question = 1 call to the LLM.** Sending it costs ~500 tokens in, and it
  sends back ~150 tokens out. (A token is about ¾ of a word — you pay by the token,
  both directions.)

  Those ~500 tokens going in are two different things glued together:

  - **System (~120 tokens)** — the rulebook. The same every single time: "only use
    the CVs below, don't make things up, cite the file, be brief." The model forgets
    everything between questions, so we re-send the rules on every call. Like
    re-reading the game rules to a goldfish before each turn.

  - **Context (~370 tokens)** — the stuff we just looked up for *this* question:
    the bits of CVs we fetched, plus the question itself. Different every time.

  - **Out (~150 tokens)** — the answer it writes back.

  **Open-ended questions cost two calls instead of one.** Before we can look
  anything up, we have to ask the LLM "what kind of question is this — a filter, one
  person, or something vague?" That's the *classifier* (or *router*) call. Then the
  second call actually answers. So you pay twice: once to figure out where to look,
  once to answer.








### Where the tokens went

| | before | after |
|---|---|---|
| routing a typical question | ~290 tok | **0** — answered from the corpus |
| a "who knows X" answer prompt | 569 tok | 366 tok (−36%) |
| reading one CV on the profile route | 442 tok | 406 tok (−8%) |
| extraction instructions, per batch | 224 tok | 148 tok (−34%) |

Two things were deliberately left alone: the grounding rules, which are what stop the
model inventing a CV, and the length cap, which is never applied to the JSON
extraction calls — enforcing it there would truncate the object mid-write.

### Two more decisions

**One database, not two.** Postgres with `pgvector` keeps rows and vectors in the same
table, so "find similar text" and "filter by years of experience" are one query
language over one set of rows. A separate vector service would be another thing to
run, another thing to keep in sync, and mostly a bill.

**The LLM cache is off by default.** Replaying a stored answer is right while tuning
prompts and wrong while serving — two users asking the same thing would get identical
bytes forever, and an outage would look healthy. Its key hashes the model and the
*full* prompt, context included, so a tuned prompt misses rather than replaying stale
output.

### What it leaves on disk

* It only ever reads `data/cvs/`
* It writes `data/extractions/` (one JSON per CV plus a hash of its text — what makes a re-ingest free)
* `data/llm_cache/` (only when `LLM_CACHE=1`) 
* For the Postgres volume, rewritten on every ingest and reset with `docker compose down -v`. Both directories are safe to delete.
