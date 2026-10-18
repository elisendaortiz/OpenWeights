"""Minimal terminal chatbot that talks to a local Ollama server.

Uses the OpenAI client pointed at Ollama's OpenAI-compatible endpoint
(http://localhost:11434/v1), which serves a local open-weights model.

How it keeps memory: the model has no memory between calls, so the script
keeps a `messages` list. Each turn it appends the user's message, sends the
whole list to the model, then appends the model's reply as an "assistant"
message. Every call therefore includes the full conversation so far, which
is what lets the model refer back to earlier turns.

Run with: uv run chat_service.py
Quit with: 'exit', 'quit' or Ctrl+C.
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # dummy value: Ollama ignores it, but the library requires one
)

MODEL = "qwen2.5:1.5b"  # gemma4:e2b-it-qat

messages = []

print(f"Chatting with {MODEL}. Type 'exit' or press Ctrl+C to quit.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    print(f"Bot: {reply}\n")
