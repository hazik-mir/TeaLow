# Provider Reference

## Google Gemini

- **Detected by**: model name containing `gemini`.
- **Base URL**: `https://generativelanguage.googleapis.com/v1beta`
- **Auth**: API key passed as a `key` query parameter.
- **Env var**: `GEMINI_API_KEY`
- **Streaming**: Server-Sent Events via `streamGenerateContent?alt=sse`.
- **Vision**: Supported via inline base64 image data or file URIs.
- **JSON mode**: Supported via `generationConfig.responseMimeType = "application/json"`.
- **Image generation**: Supported via Imagen `predict` endpoint
  (auto-selected when the model name contains `image`/`imagen`, or
  defaults to `imagen-3.0-generate-002`).

## OpenAI

- **Detected by**: model name containing `gpt-`, `o1`, `o3`, `o4`, `dall-e`, or `text-embedding`.
- **Base URL**: `https://api.openai.com/v1`
- **Auth**: `Authorization: Bearer <key>` header.
- **Env var**: `OPENAI_API_KEY`
- **Streaming**: Server-Sent Events on `/chat/completions` with `stream: true`.
- **Vision**: Supported via `image_url` content blocks (URL or base64 data URI).
- **JSON mode**: Supported via `response_format: {"type": "json_object"}`.
- **Image generation**: Supported via `/images/generations` (DALL-E models).

## Anthropic Claude

- **Detected by**: model name containing `claude`.
- **Base URL**: `https://api.anthropic.com/v1`
- **Auth**: `x-api-key` header plus `anthropic-version` header.
- **Env var**: `ANTHROPIC_API_KEY`
- **Streaming**: Server-Sent Events on `/messages` with `stream: true`,
  emitting `content_block_delta` / `message_delta` / `message_stop` events.
- **Vision**: Supported via `image` content blocks (base64 or URL source).
- **JSON mode**: Emulated by appending a system instruction requiring
  JSON-only output, then validating the response parses as JSON.
- **Image generation**: Not currently offered by Anthropic; calling
  `generate_image()` raises `ImageGenerationError`.

## DeepSeek

- **Detected by**: model name containing `deepseek`.
- **Base URL**: `https://api.deepseek.com/v1`
- **Auth**: `Authorization: Bearer <key>` header (OpenAI-compatible).
- **Env var**: `DEEPSEEK_API_KEY`
- **Streaming**: Identical wire format to OpenAI's `/chat/completions`.
- **Vision**: Supported for vision-capable DeepSeek models via the
  same `image_url` content blocks as OpenAI.
- **JSON mode**: Supported via `response_format: {"type": "json_object"}`.
- **Image generation**: Not currently offered by DeepSeek; calling
  `generate_image()` raises `ImageGenerationError`.

## Choosing a Model

TeaLow's routing (`TeaLow.client.detect_provider`) inspects the model
string for known substrings. If you use a very new or unusual model
name that doesn't match any known pattern, `detect_provider()` raises
`UnsupportedModelError`. In that case, either use a documented model
name or file an issue/PR to extend `MODEL_PREFIX_ROUTES` in
`TeaLow/constants.py`.
