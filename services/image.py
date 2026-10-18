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
from pathlib import Path

from PIL import Image, ImageOps

IMAGES_DIR = Path("data/images")  # all images live here
MAX_SIDE = 1024  # longest side in pixels after resizing (768-1024 works well)
FIRST_PROMPT = "Describe this image"

from services.config import require_env, get_client, SAMPLING_PARAMETERS


def prepare_image(data: bytes, max_side: int = MAX_SIDE) -> str:
    try:
        with Image.open(io.BytesIO(data)) as original:
            img = ImageOps.exif_transpose(original).convert("RGB")
            img.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("Not a valid image file.") from exc
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def load_image_from_data_dir(filename: str) -> bytes:
    if not filename or filename == "." or ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("Pass only the image file name, not a path.")
    path = IMAGES_DIR / filename
    if not path.resolve().is_relative_to(IMAGES_DIR.resolve()):
        raise ValueError("Image must be inside data/images/.")
    if not path.is_file():
        raise ValueError(f"Image not found: {path}")
    return path.read_bytes()


def build_image_message(data_url: str, text: str) -> dict:
    return {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}


from services.chat import stream_reply


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
    client = get_client()

    data_url = prepare_image(load_image_from_data_dir(args.image))

    messages = [build_image_message(data_url, FIRST_PROMPT)]

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

