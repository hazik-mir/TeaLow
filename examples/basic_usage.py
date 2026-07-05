"""Basic TeaLow usage: send a single message and print the response.

Run with:
    OPENAI_API_KEY=sk-... python examples/basic_usage.py
"""

from __future__ import annotations

from TeaLow import TeaLow


def main() -> None:
    # The provider (OpenAI) is detected automatically from the model name.
    # If `api` is omitted, TeaLow reads it from the OPENAI_API_KEY env var.
    ai = TeaLow(model="gpt-4o-mini")

    response = ai.send("Hello! Introduce yourself in one sentence.")
    print("Response text:", response.text)
    print("Provider:", response.provider)
    print("Model:", response.model)
    print("Usage:", response.usage)

    ai.close()


if __name__ == "__main__":
    main()
