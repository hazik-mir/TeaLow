# Architecture Overview

TeaLow is organized into clearly separated layers so that adding a new
provider or feature never requires touching unrelated code.

```
TeaLow/
├── client.py          # Synchronous TeaLow client (public entry point)
├── async_client.py     # Asynchronous AsyncTeaLow client
├── providers/
│   ├── base.py          # Abstract BaseProvider contract
│   ├── openai.py         # OpenAI adapter (also base for DeepSeek)
│   ├── deepseek.py       # DeepSeek adapter (subclasses OpenAIProvider)
│   ├── anthropic.py       # Anthropic Claude adapter
│   └── gemini.py           # Google Gemini adapter
├── session.py           # HTTPSession / AsyncHTTPSession (requests/httpx wrappers)
├── streaming.py         # StreamAccumulator for aggregating streamed chunks
├── models.py            # Message, Conversation, Response, Usage, StreamChunk
├── types.py             # TypedDicts for wire-format payloads
├── constants.py         # Provider routing table, base URLs, defaults
├── exceptions.py        # TeaLowError hierarchy
└── utils.py             # Logging, retries/backoff, env var resolution
```

## Request Lifecycle

1. **Construction** — `TeaLow(model=..., api=...)` calls
   `detect_provider()` to map the model name to a `Provider` enum,
   resolves the API key (explicit argument or environment variable),
   validates its basic shape, and instantiates the matching provider
   adapter with a shared `HTTPSession`.
2. **Conversation state** — Every call to `send()`/`stream()` appends a
   `Message` to the client's `Conversation`, which also tracks an
   optional system prompt.
3. **Provider adapter** — The adapter's `send()`/`stream()` method
   translates the provider-agnostic `Conversation` into that
   provider's wire format, issues the HTTP request through the shared
   session, and translates the response back into a normalized
   `Response` or a stream of `StreamChunk` objects.
4. **Retries** — `client.py` wraps every provider call in `retry_sync`
   (or `retry_async`), which retries on `RateLimitError` and other
   transient `TeaLowError`s using exponential backoff with jitter,
   raising `RetryExhaustedError` if all attempts fail.
5. **Error translation** — `session.py` centralizes translation of
   low-level `requests`/`httpx` exceptions and non-2xx HTTP responses
   into the `TeaLowError` hierarchy (`InvalidAPIKeyError`,
   `RateLimitError`, `APIResponseError`, etc.), so provider adapters
   never need to parse status codes themselves.

## Why a Shared Provider Contract?

`BaseProvider` defines a small, uniform surface area
(`build_headers`, `send`, `stream`, `asend`, `astream`,
`generate_image`) that every adapter implements. This is what allows
`TeaLow`/`AsyncTeaLow` to remain completely provider-agnostic: they
never branch on provider type, they just call the abstract interface.

## Sync vs. Async

`HTTPSession` (built on `requests`) and `AsyncHTTPSession` (built on
`httpx`) both perform the same error translation, so behavior is
identical whether you use `TeaLow` or `AsyncTeaLow`. Each provider
adapter accepts both a sync and an async session and picks the
appropriate one for `send/stream` vs. `asend/astream`.

## Extensibility

Adding a new provider requires:

1. A new `BaseProvider` subclass in `providers/`.
2. New entries in `constants.py` (`MODEL_PREFIX_ROUTES`,
   `DEFAULT_BASE_URLS`, `ENV_VAR_NAMES`).
3. Registration in `providers/__init__.py`.

No changes to `client.py` or `async_client.py` are required.
