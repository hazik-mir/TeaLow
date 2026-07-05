"""Demonstrates multi-turn conversation history with TeaLow.

Run with:
    ANTHROPIC_API_KEY=sk-ant-... python examples/conversation_history.py
"""

from __future__ import annotations

from TeaLow import TeaLow


def main() -> None:
    ai = TeaLow(
        model="claude-sonnet-4-6",
        system_prompt="You are a concise assistant who answers in one short sentence.",
    )

    print(ai.send("My favorite color is teal. Remember that."))
    print(ai.send("What is my favorite color?"))

    print("\nFull conversation history:")
    for message in ai.history:
        print(f"  [{message['role']}] {message['content']}")

    ai.close()


if __name__ == "__main__":
    main()
