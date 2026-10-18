"""Minimal terminal chatbot that talks to an OpenAI-compatible server.

Uses the OpenAI client pointed at any OpenAI-compatible endpoint, such as
Ollama locally (http://localhost:11434/v1) or vLLM in the cloud.

Settings come from a `.env` file (see `.env.example`):
  BASE_URL  OpenAI-compatible endpoint
  MODEL     model name served by that endpoint
  API_KEY   API key (Ollama ignores it, but the client library requires one)
Nothing is hardcoded, so the same code can point at another server just by
editing `.env`. Real environment variables take priority over the file.

How it keeps memory: the model has no memory between calls, so the script
keeps a `messages` list. Each turn it appends the user's message, sends the
whole list to the model, then appends the model's reply as an "assistant"
message. Every call therefore includes the full conversation so far, which
is what lets the model refer back to earlier turns.

Run with: uv run chat_service.py
Quit with: 'exit', 'quit' or Ctrl+C.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env into os.environ; existing env vars are not overridden


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing {name}. Copy .env.example to .env and set it.")
    return value


BASE_URL = require_env("BASE_URL")
MODEL = require_env("MODEL")
API_KEY = require_env("API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

messages = []

print(f"Chatting with {MODEL} at {BASE_URL}. Type 'exit' or press Ctrl+C to quit.\n")

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
