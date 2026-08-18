export type Meta = { type: "meta"; route: string; forced: boolean; sources: string[] };
export type Token = { type: "token"; text: string };
// Failures after the first byte arrive as an event, not an HTTP status: by then
// the 200 is already sent, so a dead stream is indistinguishable from a short
// answer unless the server says so in band.
export type Failed = { type: "error"; text: string };
export type Done = { type: "done" };
export type ChatEvent = Meta | Token | Failed | Done;

/**
 * POST /chat and hand back Server-Sent Events as they arrive.
 *
 * EventSource is not an option: it only issues GET requests, and the question
 * goes in a JSON body. So this reads the response stream directly and parses
 * the SSE framing, buffering anything incomplete — a frame is routinely split
 * across two network chunks, and parsing per-chunk drops tokens at the seam.
 */
export async function streamChat(
  question: string,
  route: string | null,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(route ? { question, route } : { question }),
    // Aborting has to reach the socket, not just stop the loop below: the
    // answer is generated as it streams, so abandoning the reader alone would
    // leave the model running — and being billed — with nobody listening.
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`chat failed: ${response.status} ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of frame.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)) as ChatEvent);
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}
