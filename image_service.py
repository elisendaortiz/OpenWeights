"""Describe an image with a vision-capable open-weights model, then chat about it.

Run from the repo root:
  uv run image_service.py image.jpg      # uses data/images/image.jpg
  uv run image_service.py other.png      # uses data/images/other.png

The image name is required. Pass only the file name: all images are assumed
to live inside data/images.

Flow:
  1. Open the image with Pillow and shrink it in memory so the longest side is
     at most MAX_SIDE pixels. The original file is never modified. 
     Notice: Text in screenshots may blur at 1024 px. If the small model misreads text,
             try raising MAX_SIDE.
  2. Encode it as JPEG, then base64, and build a data URL
     (data:image/jpeg;base64,<...>).
  3. Send it to the model in the first user message, together with a text
     prompt ("Describe this image"), and print the reply.
  4. Drop into a Q&A loop. The image is sent only in the first message, but it
     stays in the `messages` history, so later questions refer to the same image.

Settings come from a `.env` file (see `.env.example`):
  BASE_URL  OpenAI-compatible endpoint
  MODEL     model name served by that endpoint (must support vision)
  API_KEY   API key (Ollama ignores it, but the client library requires one)
Nothing is hardcoded, so the same code can point at another server just by
editing `.env`. Real environment variables take priority over the file.

Quit the Q&A loop with: 'exit', 'quit' or Ctrl+C.
"""

import argparse
import base64
import io
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGES_DIR = Path("data/images")  # all images live here
MAX_SIDE = 1024  # longest side in pixels after resizing (768-1024 works well)
FIRST_PROMPT = "Describe this image"

load_dotenv()  # reads .env into os.environ; existing env vars are not overridden


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing {name}. Copy .env.example to .env and set it.")
    return value


def image_to_data_url(path: Path, max_side: int = MAX_SIDE) -> str:
    """Load an image, shrink it in memory, and return it as a base64 data URL."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)  # respect phone-photo orientation
            img = img.convert("RGB")  # JPEG has no alpha channel (e.g. PNG screenshots)
            img.thumbnail((max_side, max_side))  # keeps aspect ratio, only shrinks
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
    except FileNotFoundError:
        raise SystemExit(f"Image not found: {path}")
    except UnidentifiedImageError:
        raise SystemExit(f"Not a valid image file: {path}")

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def stream_reply(client: OpenAI, model: str, messages: list) -> str:
    """Send the conversation, print tokens as they arrive, return the full reply."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    print("Bot: ", end="", flush=True)
    reply = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content or ""
        print(token, end="", flush=True)
        reply += token
    print("\n")
    return reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "image",
        help=f"name of the image file inside {IMAGES_DIR} (e.g. image.jpg)",
    )
    args = parser.parse_args()
    image_path = IMAGES_DIR / args.image

    base_url = require_env("BASE_URL")
    model = require_env("MODEL")
    api_key = require_env("API_KEY")
    client = OpenAI(base_url=base_url, api_key=api_key)

    data_url = image_to_data_url(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": FIRST_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    print(f"Image: {image_path} | Model: {model} at {base_url}\n")
    reply = stream_reply(client, model, messages)
    messages.append({"role": "assistant", "content": reply})

    print("Ask follow-up questions about the image. Type 'exit' or press Ctrl+C to quit.\n")
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
        reply = stream_reply(client, model, messages)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
