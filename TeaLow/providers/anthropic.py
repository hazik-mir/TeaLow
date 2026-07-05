"""Anthropic Claude provider adapter.

Implements chat completions (sync + async + streaming) and vision
input support against the Anthropic Messages API
(``https://api.anthropic.com/v1/messages``).
"""

from __future__ import annotations

import json as json_lib
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from ..exceptions import InvalidResponseError, JSONModeError
from ..models import Conversation, Message, Response, StreamChunk, Usage
from ..utils import get_logger, merge_headers
from .base import BaseProvider

logger = get_logger("providers.anthropic")

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Adapter for Anthropic's Messages API."""

    name = "anthropic"

    def build_headers(self) -> Dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        return merge_headers(headers, self.extra_headers)

    def _build_messages_payload(self, conversation: Conversation) -> List[Dict[str, Any]]:
        return [self._convert_message(message) for message in conversation.messages]

    @staticmethod
    def _convert_message(message: Message) -> Dict[str, Any]:
        if not message.images:
            return {"role": message.role, "content": message.content}

        content: List[Dict[str, Any]] = []
        for image in message.images:
            if image.get("base64"):
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image.get("mime_type", "image/png"),
                            "data": image["base64"],
                        },
                    }
                )
            elif image.get("url"):
                content.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": image["url"]},
                    }
                )
        content.append({"type": "text", "text": message.content})
        return {"role": message.role, "content": content}

    def _build_request_body(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_tokens: int,
        stream: bool,
        json_mode: bool,
        extra_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages_payload(conversation),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if conversation.system_prompt:
            body["system"] = conversation.system_prompt
        if json_mode:
            json_instruction = (
                "You must respond with valid JSON only, and nothing else."
            )
            body["system"] = (
                f"{body.get('system', '')}\n{json_instruction}".strip()
            )
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
        url = f"{self.base_url}/messages"
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
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
            content_blocks = data["content"]
            text = "".join(
                block.get("text", "") for block in content_blocks if block.get("type") == "text"
            )
            finish_reason = data.get("stop_reason")
        except (KeyError, TypeError) as exc:
            raise InvalidResponseError(
                f"Unexpected Anthropic response shape: {exc}", provider=self.name
            ) from exc

        if json_mode:
            try:
                json_lib.loads(text)
            except ValueError as exc:
                raise JSONModeError(
                    f"Anthropic response was not valid JSON: {exc}", provider=self.name
                ) from exc

        usage_raw = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("input_tokens", 0) or 0),
            completion_tokens=int(usage_raw.get("output_tokens", 0) or 0),
            total_tokens=int(usage_raw.get("input_tokens", 0) or 0)
            + int(usage_raw.get("output_tokens", 0) or 0),
        )
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
        url = f"{self.base_url}/messages"
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
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

        event_type = data.get("type")
        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            text = delta.get("text", "") or ""
            return StreamChunk(delta=text, model=self.model, provider=self.name, raw=data)
        if event_type == "message_delta":
            stop_reason = data.get("delta", {}).get("stop_reason")
            return StreamChunk(
                delta="",
                model=self.model,
                provider=self.name,
                finish_reason=stop_reason,
                raw=data,
                is_final=stop_reason is not None,
            )
        if event_type == "message_stop":
            return StreamChunk(
                delta="", model=self.model, provider=self.name, is_final=True, raw=data
            )
        return None

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
        url = f"{self.base_url}/messages"
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
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
        url = f"{self.base_url}/messages"
        body = self._build_request_body(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
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
