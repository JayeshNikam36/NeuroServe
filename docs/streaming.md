# Streaming

`"stream": true` on `/v1/chat/completions` or `/v1/completions` returns the generation as
server-sent events (`text/event-stream`) in OpenAI's chunk format, ending with `data: [DONE]`.

- With `"stream_options": {"include_usage": true}`, a last chunk with `"choices": []` carries the
  `usage`, as in OpenAI's API.
- The last chunk before `[DONE]` carries the `neuroserve` block: `e2e_ms`, `ttft_ms`, `queue_ms`,
  `generated_tokens`, and `completion_token_ids` when `neuroserve.return_token_ids` is set.
- The streamed text is exactly the text of the same request without streaming (tested), and the
  streamed tokens match Hugging Face through the official `openai` client
  (`tests/python/test_api_parity.py`).

## How a token gets to the client

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
- If the client disconnects, the request still runs to the end (cancellation is a later step).
