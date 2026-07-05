"""Streaming example: print tokens as they arrive.

Run with:
    GEMINI_API_KEY=... python examples/streaming_example.py
"""

from __future__ import annotations

from TeaLow import TeaLow


def main() -> None:
    ai = TeaLow(model="gemini-2.5-flash")

    print("Streaming response:\n")
    for chunk in ai.stream("Write a short poem about tea."):
        print(chunk.delta, end="", flush=True)
    print("\n\nDone.")

    ai.close()


if __name__ == "__main__":
    main()
