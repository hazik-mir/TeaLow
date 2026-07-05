"""Abstract base class describing the provider integration contract.

Every concrete provider (Gemini, OpenAI, Anthropic, DeepSeek) implements
this interface so that :class:`TeaLow.client.TeaLow` and
:class:`TeaLow.async_client.AsyncTeaLow` can remain provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from ..models import Conversation, Response, StreamChunk
from ..session import AsyncHTTPSession, HTTPSession


class BaseProvider(ABC):
    """Defines the operations every TeaLow provider adapter must support."""

    #: Machine-readable provider name, e.g. ``"openai"``.
    name: str = "base"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float,
        max_retries: int,
        session: HTTPSession,
        async_session: Optional[AsyncHTTPSession] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session
        self.async_session = async_session
        self.extra_headers = extra_headers or {}

    @abstractmethod
    def build_headers(self) -> Dict[str, str]:
        """Return the HTTP headers required to authenticate a request."""

    @abstractmethod
    def send(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """Send a full (non-streaming) chat request and return a Response."""

    @abstractmethod
    def stream(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_tokens: int,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[StreamChunk]:
        """Send a chat request and yield incremental :class:`StreamChunk`."""

    @abstractmethod
    async def asend(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        """Asynchronous counterpart to :meth:`send`."""

    @abstractmethod
    async def astream(
        self,
        conversation: Conversation,
        *,
        temperature: float,
        max_tokens: int,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Asynchronous counterpart to :meth:`stream`."""

    def supports_image_generation(self) -> bool:
        """Whether this provider adapter supports image generation."""
        return False

    def generate_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generate one or more images from a text prompt.

        Returns:
            A list of image URLs or Base64 strings, depending on provider.
        """
        from ..exceptions import ImageGenerationError

        raise ImageGenerationError(
            f"Provider '{self.name}' does not support image generation.",
            provider=self.name,
        )

    def validate_model(self) -> None:
        """Hook for provider-specific model-name validation. No-op by default."""
        return None
