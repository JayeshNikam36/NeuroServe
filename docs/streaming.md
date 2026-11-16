# Streaming

`"stream": true` on `/v1/chat/completions` or `/v1/completions` returns the generation as
server-sent events (`text/event-stream`) in OpenAI's chunk format, ending with `data: [DONE]`.

    data: {"id":"chatcmpl-…","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},…}]}
    data: {…"choices":[{"index":0,"delta":{"content":"Paris"},"finish_reason":null}]}
    data: {…"choices":[{"index":0,"delta":{"content":" is the capital"},"finish_reason":null}]}
    data: {…"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"neuroserve":{"ttft_ms":…,"e2e_ms":…}}
    data: [DONE]

- With `"stream_options": {"include_usage": true}`, a last chunk with `"choices": []` carries the
  `usage`, as in OpenAI's API.
- The last chunk before `[DONE]` carries the `neuroserve` block: `e2e_ms`, `ttft_ms`, `queue_ms`,
  `itl_ms`, `max_itl_ms`, `generated_tokens`, and `completion_token_ids` when
  `neuroserve.return_token_ids` is set (see docs/latency.md).
- The streamed text is exactly the text of the same request without streaming (tested), and the
  streamed tokens match Hugging Face through the official `openai` client
  (`tests/python/test_api_parity.py`).

## How a token gets to the client

    engine thread                      worker thread (GIL released)        event loop
    append_token -> request's token -> TokenStream.read() -> detokenize -> SSE chunk -> client
                    buffer (C++, own                         (never half
                    mutex + condition)                       a character)

1. The engine thread appends each new token to the request's buffer in C++
   (`RequestHandle::read_tokens`) and wakes the reader. It never calls into Python, so it never
   waits for the GIL.
2. A worker thread blocks in `TokenStream.read()` with the GIL released and gets every token not
   read yet. If the client is slower than the engine, the next read simply returns several tokens:
   the engine never waits for a reader.
3. `IncrementalDetokenizer` turns the tokens into text and holds back an incomplete UTF-8
   character until the next token completes it.
4. The event loop sends one chunk per read that produced text.

A preempted request generates its tokens again from the prompt; the buffer hands each token out
only once, and the engine logs an error if a regenerated token ever differed from the one already
streamed (greedy decoding makes them identical; see ADR-0004).

## Limits

- The access log line is written when the stream starts; a second line, `stream completed`,
  carries the usage numbers and timings.
- If the client disconnects mid-stream, the request is cancelled and its KV pages are freed at
  the next engine step (see "Cancellation" below).

## Cancellation

A request stops early in two ways:

- **The client disconnects.** A stream notices when the connection closes; a non-streaming
  request checks every 100 ms while it waits. Either way the engine request is cancelled.
- **`DELETE /v1/requests/{id}`**, where `id` is the request's `X-Request-ID` (sent by the client
  or assigned by the server and returned in the response header, so it is known as soon as the
  response starts). Answers `{"id": …, "object": "request.cancellation", "cancelled": true}`, or
  404 `request_not_found` if no request with that id is in progress.

The engine applies a cancellation at the start of its next step, before any scheduling decision:
the request leaves the queue or the running batch, moves to `CANCELLED`, and its sequence and KV
pages are freed, so they are available to that same step. A stream ends with an error event with
code `request_cancelled`; a non-streaming request gets HTTP 499 with the same code.
