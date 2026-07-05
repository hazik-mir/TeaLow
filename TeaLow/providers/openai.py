"""OpenAI provider adapter.

Implements chat completions (sync + async + streaming), vision inputs,
JSON response mode, and image generation (DALL-E) against the OpenAI
REST API (``https://api.openai.com/v1``).
"""

from __future__ import annotations

import json as json_lib
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from ..exceptions import ImageGenerationError, InvalidResponseError, JSONModeError
from ..models import Conversation, Message, Response, StreamChunk, Usage
from ..utils import get_logger, merge_headers
from .base import BaseProvider

logger = get_logger("providers.openai")

_IMAGE_GENERATION_MODELS = ("dall-e", "gpt-image")


class OpenAIProvider(BaseProvider):
    """Adapter for OpenAI's Chat Completions and Images APIs."""

    name = "openai"

    def build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return merge_headers(headers, self.extra_headers)

    def _build_messages_payload(self, conversation: Conversation) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        if conversation.system_prompt:
            payload.append({"role": "system", "content": conversation.system_prompt})
        for message in conversation.messages:
            payload.append(self._convert_message(message))
        return payload

    @staticmethod
    def _convert_message(message: Message) -> Dict[str, Any]:
        if not message.images:
            return {"role": message.role, "content": message.content}

        content: List[Dict[str, Any]] = [{"type": "text", "text": message.content}]
        for image in message.images:
            if image.get("url"):
                content.append(
                    {"type": "image_url", "image_url": {"url": image["url"]}}
                )
            elif image.get("base64"):
                mime_type = image.get("mime_type", "image/png")
                data_url = f"data:{mime_type};base64,{image['base64']}"
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        return {"role": message.role, "content": content}

    def _build_request_body(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        stream: bool,
        extra_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages_payload(conversation),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
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
        url = f"{self.base_url}/chat/completions"
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stream=False,
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
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidResponseError(
                f"Unexpected OpenAI response shape: {exc}", provider=self.name
            ) from exc

        if json_mode:
            try:
                json_lib.loads(text)
            except ValueError as exc:
                raise JSONModeError(
                    f"OpenAI response was not valid JSON: {exc}", provider=self.name
                ) from exc

        usage = Usage.from_dict(data.get("usage"))
        return Response(
            text=text,
            model=data.get("model", self.model),
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
        url = f"{self.base_url}/chat/completions"
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
            stream=True,
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
        if payload == "[DONE]":
            return StreamChunk(delta="", model=self.model, provider=self.name, is_final=True)
        try:
            data = json_lib.loads(payload)
        except ValueError:
            return None
        choices = data.get("choices") or []
        if not choices:
            return None
        delta_obj = choices[0].get("delta", {})
        delta_text = delta_obj.get("content", "") or ""
        finish_reason = choices[0].get("finish_reason")
        return StreamChunk(
            delta=delta_text,
            model=data.get("model", self.model),
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
        url = f"{self.base_url}/chat/completions"
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stream=False,
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
        url = f"{self.base_url}/chat/completions"
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
            stream=True,
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
        url = f"{self.base_url}/images/generations"
        model = self.model if any(m in self.model for m in _IMAGE_GENERATION_MODELS) else "dall-e-3"
        body: Dict[str, Any] = {"model": model, "prompt": prompt, "size": size, "n": n}
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
            items = data["data"]
        except (KeyError, TypeError) as exc:
            raise ImageGenerationError(
                f"Unexpected OpenAI image response shape: {exc}", provider=self.name
            ) from exc
        results: List[str] = []
        for item in items:
            if "url" in item:
                results.append(item["url"])
            elif "b64_json" in item:
                results.append(item["b64_json"])
        return results
