"""Vision example: send an image alongside a text prompt.

Run with:
    OPENAI_API_KEY=sk-... python examples/vision_example.py path/to/photo.jpg
"""

from __future__ import annotations

import sys

from TeaLow import TeaLow


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/vision_example.py <path-to-image>")
        sys.exit(1)

    image_path = sys.argv[1]
    ai = TeaLow(model="gpt-4o-mini")

    image = TeaLow.load_image(path=image_path)
    response = ai.send("Describe what is happening in this image.", images=[image])
    print(response)

    ai.close()


if __name__ == "__main__":
    main()
