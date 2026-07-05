# API Reference

## `TeaLow.TeaLow`

```python
TeaLow(
    model: str,
    api: str | None = None,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: float = 60.0,
    max_retries: int = 3,
    base_url: str | None = None,
    proxies: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    verify_ssl: bool = True,
    validate_key: bool = True,
)
```

### Methods

- `send(prompt, *, images=None, files=None, temperature=None, max_tokens=None, json_mode=False, remember=True, extra_params=None) -> Response`
- `stream(prompt, *, images=None, temperature=None, max_tokens=None, remember=True, extra_params=None) -> Iterator[StreamChunk]`
- `generate_image(prompt, *, size="1024x1024", n=1, extra_params=None) -> list[str]`
- `set_system_prompt(prompt: str | None) -> None`
- `reset_history() -> None`
- `close() -> None`

### Properties

- `provider: str` — the detected provider name (`"openai"`, `"gemini"`, `"anthropic"`, `"deepseek"`).
- `history: list[dict]` — the full conversation history as plain dictionaries.

### Static Helpers

- `TeaLow.load_image(path=None, url=None) -> ImagePayload`
- `TeaLow.load_file(path) -> FilePayload`

## `TeaLow.AsyncTeaLow`

Same constructor signature and semantics as `TeaLow`, except:

- `await send(...) -> Response`
- `async for chunk in stream(...)`
- `await generate_image(...) -> list[str]`
- `await close() -> None`
- Supports `async with AsyncTeaLow(...) as ai:`

## `TeaLow.Response`

```python
@dataclass
class Response:
    text: str
    model: str
    provider: str
    usage: Usage
    finish_reason: str | None
    raw: Any
    image_urls: list[str]
    image_base64: list[str]
```

`str(response)` returns `response.text`.

## `TeaLow.StreamChunk`

```python
@dataclass
class StreamChunk:
    delta: str
    model: str
    provider: str
    finish_reason: str | None
    raw: Any
    is_final: bool
```

## `TeaLow.Conversation`

- `add_user_message(content, images=None, files=None) -> Message`
- `add_assistant_message(content) -> Message`
- `set_system_prompt(prompt) -> None`
- `clear() -> None`
- `as_list(include_system=True) -> list[dict]`

## Exceptions

All exceptions inherit from `TeaLow.TeaLowError`:

- `ConfigurationError`
  - `UnsupportedModelError`
  - `MissingAPIKeyError`
- `InvalidAPIKeyError` (alias: `AuthenticationError`)
- `RateLimitError` (has `.retry_after: float | None`)
- `TimeoutError`
- `ConnectionError`
- `APIResponseError`
- `InvalidResponseError`
- `RetryExhaustedError` (has `.last_exception`, `.attempts`)
- `StreamingError`
- `FileUploadError`
- `ImageGenerationError`
- `VisionInputError`
- `JSONModeError`

## Utilities

- `TeaLow.configure_logging(level=logging.INFO, propagate=True) -> None`
- `TeaLow.detect_provider(model: str) -> Provider`
