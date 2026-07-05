"""DeepSeek provider adapter.

DeepSeek exposes an OpenAI-compatible Chat Completions API, so this
adapter subclasses :class:`OpenAIProvider` and only overrides the
behaviours that differ (provider name, lack of image generation
support, and DeepSeek-specific defaults).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..exceptions import ImageGenerationError
from .openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """Adapter for DeepSeek's OpenAI-compatible Chat Completions API."""

    name = "deepseek"

    def supports_image_generation(self) -> bool:
        return False

    def generate_image(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        n: int = 1,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        raise ImageGenerationError(
            "DeepSeek does not currently offer an image generation API.",
            provider=self.name,
        )
