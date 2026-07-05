"""Error handling example: catching TeaLow's exception hierarchy.

Run with:
    python examples/error_handling_example.py
"""

from __future__ import annotations

from TeaLow import (
    InvalidAPIKeyError,
    RateLimitError,
    TeaLow,
    TeaLowError,
    UnsupportedModelError,
)


def main() -> None:
    try:
        ai = TeaLow(model="some-unknown-model", api="fake-key-12345678")
    except UnsupportedModelError as exc:
        print("Model routing failed as expected:", exc)

    ai = TeaLow(model="gpt-4o-mini", api="clearly-not-a-real-key-12345678", max_retries=1)
    try:
        ai.send("Hello!")
    except InvalidAPIKeyError:
        print("Caught an invalid API key error, as expected with a fake key.")
    except RateLimitError as exc:
        print("Rate limited. Retry after:", exc.retry_after)
    except TeaLowError as exc:
        print("Some other TeaLow error occurred:", exc)
    finally:
        ai.close()


if __name__ == "__main__":
    main()
