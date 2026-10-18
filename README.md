# OpenWeights

A library of small, self-contained examples of how to use open-weights models for different tasks. Every example talks to the model through an OpenAI-compatible API, so the same code runs against a local [Ollama](https://ollama.com) server today and can point at another server later (for example vLLM in the cloud) by changing only a settings file.

## Local web UI

OpenWeights Playground provides **Chatbot**, **Images**, **Audio**, and **Agent** pages with separate conversations, streaming text replies, upload controls, and live time-tool results.

[Demo GIF — record `docs/ui-demo.gif`](docs/README_GIF_TODO.md)

After setup, start the UI from the repository root:

```bash
uv run streamlit run app.py
```

The browser opens at http://127.0.0.1:8501. The app binds only to `127.0.0.1`, accepts uploads up to 50 MB, and disables usage telemetry. The sidebar checks connectivity every 10 seconds on interaction. Each page has its own **New conversation** button; selecting a different upload resets that page's history.

- **Chatbot:** send a message and ask follow-up questions.
- **Images:** upload a JPG, PNG or WebP, then ask for a description. The original image message stays in history for follow-ups.
- **Audio:** upload an m4a (or MP3, WAV, OGG, AAC), click **Transcribe**, watch chunk progress, then ask questions about the transcript. Answers use the transcript, not tone or non-speech sounds.
- **Agent:** ask “What time is it in Tokyo?” to see MCP tool arguments, results, and the final answer.

UI images are processed in memory. Audio uploads use generated temporary paths; uploads and converted chunks are removed on completion or failure. UI uploads never go into `data/`. Conversations last only for the browser session. The UI starts a fresh MCP session for each agent question; the CLI keeps its session open.

**Memory:** On an 8 GB machine, close heavy apps and watch Memory Pressure in Activity Monitor. Keep conversations and recordings short; start a new conversation to release old context. Use only your existing model.

The UI, like the CLIs, talks only to `BASE_URL`. Changing `.env` to a compatible vLLM server requires no code changes. The configured model/server must support the selected input type; the text-only default in `.env.example` cannot handle image or audio requests.

The UI dependency requires Pillow below 13; this branch uses Pillow 12.x with the same image conversion and resizing behavior.

### Tests

```bash
uv run pytest
```

Tests cover image processing, file validation, audio splitting and cleanup, agent cancellation, and UI interactions using mocked model responses.

## Use cases

| # | Use case | Script | What it does | Model must support |
| --- | --- | --- | --- | --- |
| 1 | [Chatbots](#use-case-1-chatbots) | `chat_service.py`, `chat_streaming_service.py` | Terminal chat that remembers the conversation | text |
| 2 | [Image](#use-case-2-image) | `image_service.py` | Describe an image, then ask follow-up questions about it | vision |
| 3 | [Audio](#use-case-3-audio) | `audio_service.py` | Transcribe a recording, then ask questions about the whole audio | audio |
| 4 | [Agent (time)](#use-case-4-agent-time) | `agent_timezone_service.py` | Chat where the model answers time questions by calling tools on an MCP server | tools |

## Getting started

### Requirements

- **Python 3.13 or newer** and [uv](https://docs.astral.sh/uv/) to manage the environment and dependencies.
- **Ollama** running locally, with at least one model pulled (see [Models](#models)). Recent models may need a recent Ollama version. `ollama show <model>` prints the minimum under `requires`.
- **ffmpeg**, only for the audio use case. `uv` cannot install it because it is a system program, not a Python package.
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`
  - Check it with `ffmpeg -version`.
- **uvx**, only for the agent use case. It comes with uv, so there is nothing extra to install. It downloads and runs the MCP time server on demand, so the first run needs internet access. Check it with `uvx --version`.

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
- **Agent** needs a model with the `tools` capability (tool calling). Both `qwen2.5:1.5b` and `gemma4:e2b-it-qat` have it.

`gemma4:e2b-it-qat` has vision, audio and tools, so it covers use cases 2 to 4. Check what any model supports with:

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
- Text in screenshots can blur when the image is shrunk to 1024 px. If the model misreads text, raise `MAX_SIDE` in `services/image.py`.
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

## Use case 4: Agent (time)

A chat where the model answers time questions by calling tools instead of guessing. The tools come from an [MCP](https://modelcontextprotocol.io) server, `mcp-server-time`, which provides `get_current_time` and `convert_time`. `agent_timezone_service.py` is the agent: it connects the model to the tools and runs the loop between them.

**How it works:**

1. Starts the MCP time server (`uvx mcp-server-time`) as a subprocess and talks to it over stdin/stdout with the MCP Python client. It lists the server's tools and prints their names at startup.
2. Converts each MCP tool to the OpenAI `tools` format (name, description, JSON schema) and sends the list with every request. A short system prompt tells the model to use the tools for any time question and to use IANA timezone names such as `Asia/Tokyo`.
3. If the reply contains `tool_calls`, runs each call through MCP, appends the assistant message and a `role: "tool"` message with the result (matching `tool_call_id`), and calls the model again. The model then writes the final answer.
4. Repeats step 3 until the model replies without a tool call, with a cap of 5 rounds per question so a confused model cannot loop forever. This loop is what makes it an agent.
5. If a tool call fails (unknown tool, invalid JSON arguments, an error from the server), the error is sent back to the model as the tool result so it can retry.
6. Keeps the full `messages` history, so follow-ups work ("and in New York?", "what's the difference between them?").

Every tool call and its result is printed, for example:

```
→ get_current_time(timezone="Asia/Tokyo")
  ← {"timezone": "Asia/Tokyo", ...}
```

### Run

```bash
uv run python agent_timezone_service.py
```

Type `exit` or `quit`, or press Ctrl+C, to leave. Try easy questions first ("What time is it in Tokyo?"), then harder ones ("convert 3pm Singapore to London time", "what time is it in the capital of Peru?"), and check each answer against a real clock.

### Adding more tools

The MCP servers are listed in `SERVERS` in `services/agent.py`. To give the agent more abilities, such as stock market data, add one more entry there. Their tools are merged into one list and each call is routed to the server that owns the tool. The loop stays the same.

### Tips and limits

- Small models are unreliable at tool calling. Watch the `→` lines to see when the model picks the wrong tool, mangles a timezone name (for example `Tokyo` instead of `Asia/Tokyo`) or ignores the tools.
- `(no tool used)` means the model answered without calling a tool. That is fine for a follow-up that needs no tool, but a problem for a plain time question.
- `MODEL` must support tool calling. Check for `tools` under capabilities with `ollama show <model>`.
- Running `uvx mcp-server-time` by hand prints JSON validation errors. That is expected: the server speaks JSON over stdin/stdout and waits for an MCP client, not a human. Stop it with Ctrl+C. The script launches it for you.

## Using another server (for example vLLM)

Because the scripts only use the OpenAI-compatible API, moving to another server means editing `.env` and nothing else:

```
BASE_URL=https://your-vllm-host/v1
MODEL=<the model name your server serves>
API_KEY=<your key>
```

The server and model must support whatever the use case needs (text, vision, audio or tool calling).

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `Missing BASE_URL` (or `MODEL`, `API_KEY`) | `.env` doesn't exist or is missing the variable. Run `cp .env.example .env`. |
| Connection error | Ollama isn't running, or `BASE_URL` is wrong. |
| `invalid model name` / 400 | `MODEL` doesn't exactly match a pulled model, or has a stray `#comment` glued to it (see Configuration). Check with `ollama list`. |
| Image or audio ignored, or the request is rejected | `MODEL` lacks the `vision` or `audio` capability. Check with `ollama show <model>`. |
| `Image not found` / `<name> not found in data/audio/` | The file isn't in `data/images/` or `data/audio/`, or you passed a path instead of just the name. |
| `ffmpeg not found` | Install it (see Requirements) and check with `ffmpeg -version`. |
| `uvx not found` / `Could not start MCP server 'time'` | `uvx` comes with uv, so check your uv install. The first run also needs internet access to download `mcp-server-time`. |
| Agent: "The model server returned an error" | `MODEL` probably doesn't support tool calling. Check for `tools` under capabilities with `ollama show <model>`. |
