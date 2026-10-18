# OpenWeights

A library of small, self-contained examples of how to use open-weights models for different tasks. Every example talks to the model through an OpenAI-compatible API, so the same code runs against a local [Ollama](https://ollama.com) server today and can point at another server later (for example vLLM in the cloud) by changing only a settings file.

More use cases will be added over time.

## Use cases

| # | Use case | Script | What it does | Model must support |
| --- | --- | --- | --- | --- |
| 1 | [Chatbots](#use-case-1-chatbots) | `chat_service.py`, `chat_streaming_service.py` | Terminal chat that remembers the conversation | text |
| 2 | [Image](#use-case-2-image) | `image_service.py` | Describe an image, then ask follow-up questions about it | vision |
| 3 | [Audio](#use-case-3-audio) | `audio_service.py` | Transcribe a recording, then ask questions about the whole audio | audio |

## Getting started

### Requirements

- **Python 3.13 or newer** and [uv](https://docs.astral.sh/uv/) to manage the environment and dependencies.
- **Ollama** running locally, with at least one model pulled (see [Models](#models)). Recent models may need a recent Ollama version. `ollama show <model>` prints the minimum under `requires`.
- **ffmpeg**, only for the audio use case. `uv` cannot install it because it is a system program, not a Python package.
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`
  - Check it with `ffmpeg -version`.

### Install

```bash
uv sync                      # create .venv and install dependencies
cp .env.example .env         # then edit .env (see Configuration)
```

Run every script from the repo root with `uv run`, so the relative paths (`data/...`) resolve correctly.

### Configuration

All settings live in `.env` (copied from `.env.example`). Nothing is hardcoded in the scripts.

| Variable | Meaning |
| --- | --- |
| `BASE_URL` | OpenAI-compatible endpoint. For local Ollama: `http://localhost:11434/v1` |
| `MODEL` | Name of the model served at that endpoint, exactly as Ollama lists it |
| `API_KEY` | API key. Ollama ignores it, but the client library requires a non-empty value, so `ollama` works |

Notes:

- `.env` is gitignored. Commit `.env.example`, never `.env`.
- Real environment variables take priority over `.env`.
- Don't put a comment on the same line as a value without a space before the `#`. In `MODEL=a#b` the `#b` becomes part of the value and the server answers "invalid model name". Put alternatives on their own commented line instead:

  ```
  MODEL=gemma4:e2b-it-qat
  # MODEL=qwen2.5:1.5b
  ```

### Models

The use cases need different model capabilities, and all scripts read the same `MODEL`:

- **Chatbots** work with any chat model, including small text-only ones such as `qwen2.5:1.5b` (the default in `.env.example`).
- **Image** needs a model with the `vision` capability.
- **Audio** needs a model with the `audio` capability.

`gemma4:e2b-it-qat` has both, so it is the model to use for use cases 2 and 3. Check what any model supports with:

```bash
ollama show gemma4:e2b-it-qat      # look under "Capabilities"
ollama pull gemma4:e2b-it-qat      # if you don't have it yet
```

### Data and outputs

Your own images, recordings and transcripts are not part of the repo. The folder structure under `data/` is part of the repo (each folder holds an empty `.gitkeep` placeholder), so the folders exist right after cloning. Everything you put inside them, and everything the scripts generate there, is gitignored and never tracked:

```
data/
  images/     # input images for the image use case
  audio/      # input recordings for the audio use case (temp files go in audio/tmp/ and are deleted)
  outputs/    # generated transcripts, one <name>.txt per recording (created automatically)
```

## Use case 1: Chatbots

A terminal chatbot that talks to an open-weights model.

| Script | What it does |
| --- | --- |
| `chat_service.py` | Chat loop that keeps the conversation history. The reply appears all at once. |
| `chat_streaming_service.py` | Same chat loop, but tokens are printed as they arrive, like a real chat. |

**How it remembers the conversation:** the model has no memory between calls. The script keeps a `messages` list, appends each user message, sends the whole list, and appends the model's reply. Every request therefore carries the full conversation, which is what lets the model refer back to earlier turns.

**How streaming works:** the request uses `stream=True`, so the reply arrives as small chunks. Each token is printed as soon as it arrives and the chunks are joined into the full reply, which is then stored in `messages`.

### Run

```bash
uv run chat_service.py
uv run chat_streaming_service.py
```

Type `exit` or `quit`, or press Ctrl+C, to leave.

## Use case 2: Image

Describe an image with a vision model, then ask follow-up questions about it. `image_service.py` prints a description first, then drops into a Q&A loop.

**How it works:**

1. Opens the image with Pillow and shrinks it in memory so the longest side is at most 1024 px. The original file is never modified. Phone-photo orientation is respected, and images with transparency (such as PNG screenshots) are converted to RGB.
2. Encodes it as JPEG, then base64, and builds a data URL (`data:image/jpeg;base64,...`).
3. Sends it in the first user message, together with the text prompt "Describe this image", and prints the reply.
4. Keeps asking for questions. The image is sent only in the first message, but it stays in the `messages` history, so later questions refer to the same image.

### Run

Put the image in `data/images/` first, then pass only the file name, including its extension. The name is required and there is no default image.

```bash
uv run image_service.py image.jpg
uv run image_service.py screenshot.png
```

Don't pass a folder path such as `data/images/image.jpg`. The script adds `data/images/` itself, so that would not be found.

Type `exit` or `quit`, or press Ctrl+C, to leave.

### Tips and limits

- Small models make mistakes on images. Try a photo, a screenshot and an image with text to see where yours struggles.
- Text in screenshots can blur when the image is shrunk to 1024 px. If the model misreads text, raise `MAX_SIDE` at the top of `image_service.py`.
- `MODEL` must support vision. Otherwise the server rejects the request or ignores the image.

## Use case 3: Audio

Transcribe a recording with an audio model, then ask questions about the whole audio. `audio_service.py` transcribes first and answers questions afterwards.

**How it works:**

1. Converts the file to 16 kHz, mono, 16-bit WAV with ffmpeg, in a temp folder named after the input (`data/audio/tmp/<name>/`). The original file is never modified.
2. Splits the WAV into 25-second chunks (`chunk_000.wav`, `chunk_001.wav`, ...). The margin below the model's 30 s limit avoids cut-offs. The cuts are plain fixed-length ones with no overlap, so a word at a boundary can be sliced in half.
3. Transcribes the chunks in order. Each chunk is a separate, independent request (base64 WAV sent as an `input_audio` item), and earlier audio is never resent. Each transcript is appended to `data/outputs/<name>.txt` as it arrives, so a crash does not lose progress.
4. Joins the transcripts in order and deletes the temp files.
5. Asks "What would you like to know about this audio?" and sends one text-only request containing the full transcript plus your question. The transcript stays in the `messages` history, so follow-up questions have the whole audio to work with.

If transcription fails, for example because the server rejects the audio, the script prints the server's error and keeps the temp files so you can inspect them.

### Run

Put the recording in `data/audio/` first, then pass only the file name, including its extension. The name is required and there is no default file. Folder parts such as `../x.m4a` are rejected, so only files inside `data/audio/` can be read.

```bash
uv run python audio_service.py pigs.m4a
```

The transcript is saved to `data/outputs/pigs.txt`. Type `exit` or `quit`, or press Ctrl+C, to leave.

### Tips and limits

- Try a 1-minute clip first, then a longer one, and check that chunk boundaries don't garble words.
- The server behind `BASE_URL` must accept audio input (`input_audio`) and `MODEL` must support audio. If your server rejects it, point `.env` at one that does, for example llama-server. The code stays the same.

## Using another server (for example vLLM)

Because the scripts only use the OpenAI-compatible API, moving to another server means editing `.env` and nothing else:

```
BASE_URL=https://your-vllm-host/v1
MODEL=<the model name your server serves>
API_KEY=<your key>
```

The server and model must support whatever the use case needs (text, vision or audio).

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `Missing BASE_URL` (or `MODEL`, `API_KEY`) | `.env` doesn't exist or is missing the variable. Run `cp .env.example .env`. |
| Connection error | Ollama isn't running, or `BASE_URL` is wrong. |
| `invalid model name` / 400 | `MODEL` doesn't exactly match a pulled model, or has a stray `#comment` glued to it (see Configuration). Check with `ollama list`. |
| Image or audio ignored, or the request is rejected | `MODEL` lacks the `vision` or `audio` capability. Check with `ollama show <model>`. |
| `Image not found` / `<name> not found in data/audio/` | The file isn't in `data/images/` or `data/audio/`, or you passed a path instead of just the name. |
| `ffmpeg not found` | Install it (see Requirements) and check with `ffmpeg -version`. |
