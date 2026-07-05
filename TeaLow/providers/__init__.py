"""Provider adapters for the TeaLow SDK.

Each module in this package implements :class:`TeaLow.providers.base.BaseProvider`
for a specific AI vendor. Use :func:`get_provider_class` to resolve the
appropriate adapter class from a :class:`TeaLow.constants.Provider` value.
"""

from __future__ import annotations

from typing import Dict, Type

from ..constants import Provider
from .anthropic import AnthropicProvider
from .base import BaseProvider
from .deepseek import DeepSeekProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

_PROVIDER_CLASSES: Dict[Provider, Type[BaseProvider]] = {
    Provider.GEMINI: GeminiProvider,
    Provider.OPENAI: OpenAIProvider,
    Provider.ANTHROPIC: AnthropicProvider,
    Provider.DEEPSEEK: DeepSeekProvider,
}


def get_provider_class(provider: Provider) -> Type[BaseProvider]:
    """Return the concrete provider adapter class for a given provider enum."""
    return _PROVIDER_CLASSES[provider]


__all__ = [
    "BaseProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "DeepSeekProvider",
    "get_provider_class",
]
