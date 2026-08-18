import { useEffect, useRef, useState } from "react";

import { streamChat } from "./api";

type Message = {
  role: "user" | "assistant";
  text: string;
  route?: string;
  forced?: boolean;
  sources?: string[];
  failed?: boolean;
  stopped?: boolean;
};

const SAMPLES = [
  "Who has experience with Python?",
  "Which candidate graduated from UPC?",
  "Who seems strongest at leading teams?",
];

const MAX_COMPOSER_HEIGHT = 200;

/**
 * Failures that never reach the backend, so the backend cannot name them.
 *
 * The server explains its own errors in the stream; these are the ones that
 * happen before or instead of a response — the API being down is the common
 * one in development, and "failed to fetch" alone sends people hunting through
 * the wrong logs.
 */
function describe(caught: unknown): string {
  if (caught instanceof TypeError) {
    return "Could not reach the API. Is `make api` running on :8000?";
  }
  if (caught instanceof Error) {
    return caught.message.startsWith("chat failed:")
      ? `The API rejected the request — ${caught.message.replace("chat failed: ", "")}.`
      : `Unknown error: ${caught.message}`;
  }
  return `Unknown error: ${String(caught)}`;
}

export function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  // Forcing the naive path is the whole demo: the same aggregation question
  // answered by SQL and by top-k, side by side.
  const [forceNaive, setForceNaive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stageRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  // Held in a ref rather than state: stopping must not wait for a re-render,
  // and the controller is not something the UI draws.
  const inFlight = useRef<AbortController | null>(null);

  function stop() {
    inFlight.current?.abort();
  }

  const introducing = messages.length === 0;

  // Pin to the newest token. scrollTop is assigned directly rather than going
  // through scrollIntoView because this runs once per streamed token, and a
  // smooth scroll queues an animation per token that never catches up.
  useEffect(() => {
    const stage = stageRef.current;
    if (stage) stage.scrollTop = stage.scrollHeight;
  }, [messages]);

  // The composer grows with its content up to a cap, after which it scrolls.
  // A textarea cannot do this in CSS alone: height has to be measured.
  useEffect(() => {
    const composer = composerRef.current;
    if (!composer) return;
    composer.style.height = "auto";
    composer.style.height = `${Math.min(composer.scrollHeight, MAX_COMPOSER_HEIGHT)}px`;
  }, [question]);

  async function ask(text: string) {
    if (!text.trim() || busy) return;
    setBusy(true);
    setError(null);
    setQuestion("");
    setMessages((m) => [...m, { role: "user", text }, { role: "assistant", text: "" }]);

    // Only ever mutates the last message, which is the assistant turn just added.
    const patch = (change: Partial<Message>) =>
      setMessages((m) => {
        const next = [...m];
        next[next.length - 1] = { ...next[next.length - 1], ...change };
        return next;
      });

    const controller = new AbortController();
    inFlight.current = controller;

    try {
      await streamChat(
        text,
        forceNaive ? "semantic" : null,
        (event) => {
          if (event.type === "meta") {
            patch({ route: event.route, forced: event.forced, sources: event.sources });
          } else if (event.type === "error") {
            // Onto the turn itself as well as the banner: the banner sits by
            // the composer and says nothing about which question died, and an
            // assistant turn left with no text shows its dots forever.
            patch({ failed: true });
            setError(event.text);
          } else if (event.type === "token") {
            setMessages((m) => {
              const next = [...m];
              const last = next[next.length - 1];
              next[next.length - 1] = { ...last, text: last.text + event.text };
              return next;
            });
          }
        },
        controller.signal,
      );
    } catch (caught) {
      // Stopping is a thing the user did, not a thing that went wrong, so it
      // gets no red banner — just a note that the answer is incomplete.
      // Matched on `name` rather than `instanceof DOMException`, which is not
      // what every runtime throws for an aborted fetch.
      if (caught instanceof Error && caught.name === "AbortError") {
        patch({ stopped: true });
      } else {
        patch({ failed: true });
        setError(describe(caught));
      }
    } finally {
      inFlight.current = null;
      setBusy(false);
      composerRef.current?.focus();
    }
  }

  return (
    <div className={introducing ? "app app--intro" : "app"}>
      <header className="topbar">
        <div className="brand">
          <span className="brand__mark" aria-hidden="true">
            <svg viewBox="0 0 16 16" width="13" height="13">
              <rect x="1" y="1" width="6" height="6" fill="currentColor" />
              <rect x="9" y="1" width="6" height="6" fill="currentColor" opacity="0.55" />
              <rect x="1" y="9" width="6" height="6" fill="currentColor" opacity="0.55" />
              <rect x="9" y="9" width="6" height="6" fill="currentColor" opacity="0.28" />
            </svg>
          </span>
          CV Screening
        </div>

        <label className="switch" title="Bypass routing and answer with plain top-k retrieval">
          <input
            type="checkbox"
            checked={forceNaive}
            onChange={(e) => setForceNaive(e.target.checked)}
          />
          <span className="switch__track" aria-hidden="true" />
          Force plain top-k
        </label>
      </header>

      <div className="stage" ref={stageRef}>
        {introducing ? (
          <div className="intro">
            <h1>Query the candidate corpus</h1>
            <p>
              Thirty CVs, indexed and searchable. Aggregations run as SQL, single profiles
              resolve by name, everything else falls back to semantic search.
            </p>
          </div>
        ) : (
          <ol className="thread">
            {messages.map((message, i) => (
              <li key={i} className={`turn turn--${message.role}`}>
                <div className="turn__body">
                  {message.role === "assistant" && (
                    <div className="turn__meta">
                      <span className="turn__role">Response</span>
                      {message.route && (
                        <span className={`route route--${message.route}`}>
                          {message.route}
                          {message.forced ? " · forced" : ""}
                        </span>
                      )}
                    </div>
                  )}

                  {message.text ? (
                    <div className="turn__text">
                      {message.text}
                      {message.stopped && <span className="turn__note"> — stopped</span>}
                    </div>
                  ) : message.failed ? (
                    <div className="turn__text turn__text--quiet">No answer — see below.</div>
                  ) : message.stopped ? (
                    <div className="turn__text turn__text--quiet">Stopped before any answer.</div>
                  ) : (
                    message.role === "assistant" && (
                      <span className="working" aria-label="Working">
                        <i />
                        <i />
                        <i />
                      </span>
                    )
                  )}

                  {message.sources && message.sources.length > 0 && (
                    <div className="sources">
                      <span className="sources__label">
                        {message.sources.length === 1
                          ? "1 CV used"
                          : `${message.sources.length} CVs used`}
                      </span>
                      <ul>
                        {message.sources.map((source) => (
                          <li key={source}>
                            {/* Proxied to the backend's /cvs mount, so the
                                reader can open the document the answer cites. */}
                            <a
                              href={`/api/cvs/${encodeURIComponent(source)}`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              {source}
                            </a>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ol>
        )}
      </div>

      <div className="dock">
        {error && <p className="error">{error}</p>}

        <form
          className={busy ? "composer composer--busy" : "composer"}
          onSubmit={(e) => {
            e.preventDefault();
            ask(question);
          }}
        >
          <textarea
            ref={composerRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              // Enter sends, Shift+Enter breaks the line — the convention every
              // chat interface uses, and the reason this is a textarea.
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask(question);
              }
            }}
            placeholder="Ask a question about the candidates"
            rows={1}
            disabled={busy}
            autoFocus
          />
          {/* The same slot becomes Stop while streaming, the way every chat
              interface does it — a Send button that is merely disabled leaves
              the reader with no way out of a long answer. */}
          {busy ? (
            <button type="button" className="stop" onClick={stop} title="Stop generating">
              <svg viewBox="0 0 16 16" width="11" height="11" aria-hidden="true">
                <rect x="3" y="3" width="10" height="10" rx="1.5" fill="currentColor" />
              </svg>
              Stop
            </button>
          ) : (
            <button type="submit" disabled={!question.trim()}>
              Send
            </button>
          )}
        </form>

        {introducing && (
          <div className="samples">
            {SAMPLES.map((sample) => (
              <button key={sample} onClick={() => ask(sample)} disabled={busy}>
                {sample}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
