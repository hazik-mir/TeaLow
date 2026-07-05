"""JSON response mode example.

Run with:
    OPENAI_API_KEY=sk-... python examples/json_mode_example.py
"""

from __future__ import annotations

import json

from TeaLow import TeaLow


def main() -> None:
    ai = TeaLow(model="gpt-4o-mini")

    response = ai.send(
        "Return a JSON object describing a fictional character with keys "
        "'name', 'age', and 'occupation'. Respond with JSON only.",
        json_mode=True,
    )

    data = json.loads(response.text)
    print("Parsed JSON:", data)

    ai.close()


if __name__ == "__main__":
    main()
