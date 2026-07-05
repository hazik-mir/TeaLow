"""Image generation example.

Run with:
    OPENAI_API_KEY=sk-... python examples/image_generation_example.py
"""

from __future__ import annotations

from TeaLow import TeaLow


def main() -> None:
    ai = TeaLow(model="dall-e-3")

    urls = ai.generate_image(
        "A minimalist watercolor illustration of a teapot on a low wooden table",
        size="1024x1024",
        n=1,
    )
    for url in urls:
        print(url)

    ai.close()


if __name__ == "__main__":
    main()
