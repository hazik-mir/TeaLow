"""Google Gemini provider adapter.

Implements chat generation (sync + async + streaming), vision inputs,
JSON response mode, and image generation against the Google Generative
Language API (``https://generativelanguage.googleapis.com/v1beta``).
"""

from __future__ import annotations

import json as json_lib
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from ..exceptions import ImageGenerationError, InvalidResponseError, JSONModeError
from ..models import Conversation, Message, Response, StreamChunk, Usage
from ..utils import get_logger, merge_headers
from .base import BaseProvider

logger = get_logger("providers.gemini")

_IMAGE_MODEL_HINTS = ("image", "imagen")


class GeminiProvider(BaseProvider):
    """Adapter for Google's Gemini ``generateContent`` API."""

    name = "gemini"

    def build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        return merge_headers(headers, self.extra_headers)

    def _endpoint(self, method: str) -> str:
        return f"{self.base_url}/models/{self.model}:{method}?key={self.api_key}"

    @staticmethod
    def _convert_role(role: str) -> str:
        return "model" if role == "assistant" else "user"

    def _convert_message(self, message: Message) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = [{"text": message.content}]
        for image in message.images:
            if image.get("base64"):
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": image.get("mime_type", "image/png"),
                            "data": image["base64"],
                        }
                    }
                )
            elif image.get("url"):
                parts.append({"file_data": {"file_uri": image["url"]}})
        return {"role": self._convert_role(message.role), "parts": parts}

    def _build_request_body(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        extra_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        contents = [self._convert_message(m) for m in conversation.messages]
        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if conversation.system_prompt:
            body["systemInstruction"] = {
                "parts": [{"text": conversation.system_prompt}]
            }
        if extra_params:
            body.update(extra_params)
        return body

    def send(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        url = self._endpoint("generateContent")
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            extra_params=extra_params,
        )
        http_response = self.session.request(
            "POST",
            url,
            headers=self.build_headers(),
            json=body,
            timeout=self.timeout,
            provider=self.name,
        )
        data = http_response.json()
        return self._parse_response(data, json_mode=json_mode)

    def _parse_response(self, data: Dict[str, Any], *, json_mode: bool) -> Response:
        try:
            candidate = data["candidates"][0]
            parts = candidate["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
            finish_reason = candidate.get("finishReason")
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidResponseError(
                f"Unexpected Gemini response shape: {exc}", provider=self.name
            ) from exc

        if json_mode:
            try:
                json_lib.loads(text)
            except ValueError as exc:
                raise JSONModeError(
                    f"Gemini response was not valid JSON: {exc}", provider=self.name
                ) from exc

        usage_raw = data.get("usageMetadata") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("promptTokenCount", 0) or 0),
            completion_tokens=int(usage_raw.get("candidatesTokenCount", 0) or 0),
            total_tokens=int(usage_raw.get("totalTokenCount", 0) or 0),
        )
        return Response(
            text=text,
            model=self.model,
            provider=self.name,
            usage=usage,
            finish_reason=finish_reason,
            raw=data,
        )

    def stream(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_tokens: int,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[StreamChunk]:
        url = self._endpoint("streamGenerateContent") + "&alt=sse"
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
            extra_params=extra_params,
        )
        for line in self.session.stream_lines(
            "POST",
            url,
            headers=self.build_headers(),
            json=body,
            timeout=self.timeout,
            provider=self.name,
        ):
            chunk = self._parse_sse_line(line)
            if chunk is not None:
                yield chunk

    def _parse_sse_line(self, line: str) -> Optional[StreamChunk]:
        if not line.startswith("data:"):
            return None
        payload = line[len("data:") :].strip()
        if not payload:
            return None
        try:
            data = json_lib.loads(payload)
        except ValueError:
            return None
        try:
            candidate = data["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
            finish_reason = candidate.get("finishReason")
        except (KeyError, IndexError, TypeError):
            return None
        return StreamChunk(
            delta=text,
            model=self.model,
            provider=self.name,
            finish_reason=finish_reason,
            raw=data,
            is_final=finish_reason is not None,
        )

    async def asend(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        assert self.async_session is not None
        url = self._endpoint("generateContent")
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            extra_params=extra_params,
        )
        http_response = await self.async_session.request(
            "POST",
            url,
            headers=self.build_headers(),
            json=body,
            timeout=self.timeout,
            provider=self.name,
        )
        data = http_response.json()
        return self._parse_response(data, json_mode=json_mode)

    async def astream(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_tokens: int,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        assert self.async_session is not None
        url = self._endpoint("streamGenerateContent") + "&alt=sse"
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
            extra_params=extra_params,
        )
        async for line in self.async_session.stream_lines(
            "POST",
            url,
            headers=self.build_headers(),
            json=body,
            timeout=self.timeout,
            provider=self.name,
        ):
            chunk = self._parse_sse_line(line)
            if chunk is not None:
                yield chunk

    def supports_image_generation(self) -> bool:
        return True

    def generate_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        model = self.model if any(h in self.model for h in _IMAGE_MODEL_HINTS) else "imagen-3.0-generate-002"
        url = f"{self.base_url}/models/{model}:predict?key={self.api_key}"
        body: Dict[str, Any] = {
            "instances": [{"prompt": prompt}],
            "parameters": {"sampleCount": n},
        }
        if extra_params:
            body.update(extra_params)
        http_response = self.session.request(
            "POST",
            url,
            headers=self.build_headers(),
            json=body,
            timeout=self.timeout,
            provider=self.name,
        )
        data = http_response.json()
        try:
            predictions = data["predictions"]
        except (KeyError, TypeError) as exc:
            raise ImageGenerationError(
                f"Unexpected Gemini image response shape: {exc}", provider=self.name
            ) from exc
        results: List[str] = []
        for prediction in predictions:
            image_b64 = prediction.get("bytesBase64Encoded")
            if image_b64:
                results.append(image_b64)
        return results
