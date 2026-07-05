"""Async usage example: concurrent requests with AsyncTeaLow.

Run with:
    OPENAI_API_KEY=sk-... python examples/async_example.py
"""

from __future__ import annotations

import asyncio

from TeaLow import AsyncTeaLow


async def ask_one(ai: AsyncTeaLow, question: str) -> None:
    response = await ai.send(question, remember=False)
    print(f"Q: {question}\nA: {response}\n")


async def main() -> None:
    ai = AsyncTeaLow(model="gpt-4o-mini")

    questions = [
        "What is the boiling point of water in Celsius?",
        "Name one country that produces a lot of tea.",
        "What color is chlorophyll?",
    ]

    # Fire off all requests concurrently.
    await asyncio.gather(*(ask_one(ai, q) for q in questions))

    print("Streaming example:")
    async for chunk in ai.stream("Say a friendly one-line goodbye."):
        print(chunk.delta, end="", flush=True)
    print()

    await ai.close()


if __name__ == "__main__":
    asyncio.run(main())
