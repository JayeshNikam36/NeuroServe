# API errors

Every error response uses OpenAI's format and carries an `X-Request-ID` header:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "max_tokens", "code": null}}
```

Search the server log for the request id to find the full context (and traceback, for 500s).

With `"stream": true`, errors found before the stream starts (invalid request, prompt too long)
are normal HTTP errors as below. An error after the stream has started cannot
change the status (200 is already sent): the stream ends with one event
`data: {"error": {...}}` in the same format, and no `data: [DONE]`.

| Status | `type` | `code` | When |
| --- | --- | --- | --- |
| 400 | `invalid_request_error` | `null` | Invalid body: missing or wrong-typed field, empty messages/prompt, bad role, token id outside the vocabulary, `max_tokens` above the limit (`param` names the field) |
| 400 | `invalid_request_error` | `null` | `stream_options` without `stream=true` |
| 400 | `invalid_request_error` | `unsupported_parameter` | `n>1`, or `temperature>0` (not supported yet) |
| 404 | `invalid_request_error` | `model_not_found` | `model` is not the served model |
| 404 | `invalid_request_error` | `null` | Unknown URL |
| 413 | `invalid_request_error` | `context_length_exceeded` | Prompt longer than `engine.max_prompt_tokens` |
| 500 | `server_error` | `internal_error` | Unexpected server error (message contains the request id; details only in the log) |
| 500 | `server_error` | `gpu_error` | CUDA error during generation; the server marks itself unhealthy |
| 503 | `server_error` | `kv_cache_exhausted` | Not enough free KV-cache pages for the request |
| 503 | `server_error` | `service_unavailable` | Every request after a GPU error (`/health` also returns 503; restart required), or a request caught by an engine shutdown |
