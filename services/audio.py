"""Transcribe an audio file with an audio-capable open-weights model, then chat about it.

Run from the repo root, passing only the file name (audio files always live
in data/audio/):
  uv run python audio_service.py pigs.m4a

Flow:
  1. Convert the file to 16 kHz, mono, 16-bit PCM WAV with ffmpeg, inside a
     temp folder named after the input (data/audio/tmp/<name>/). The original
     file is never modified.
  2. Split the WAV into 25-second chunks (chunk_000.wav, chunk_001.wav, ...).
     The margin below the model's 30 s limit avoids cut-offs. Chunks are plain
     fixed-length cuts, so a word at a boundary can be sliced in half.
  3. Transcribe the chunks in order. Each chunk is a separate, independent
     request (base64 WAV sent as an `input_audio` item). The transcript is
     appended to data/outputs/<name>.txt as it goes, so a crash does not lose progress.
  4. Join the transcripts and delete the temp files.
  5. Ask what you want to know about the audio, then send one text-only request
     with the full transcript plus your question. Keep chatting: the transcript
     stays in the `messages` history, so follow-up questions have the whole audio.

Requires ffmpeg on the system (brew install ffmpeg / apt install ffmpeg).

Settings come from a `.env` file (see `.env.example`):
  BASE_URL  OpenAI-compatible endpoint (must accept `input_audio`)
  MODEL     model name served by that endpoint (must support audio)
  API_KEY   API key (Ollama ignores it, but the client library requires one)
Nothing is hardcoded, so the same code can point at another server just by
editing `.env`. Real environment variables take priority over the file.

Quit the Q&A loop with: 'exit', 'quit' or Ctrl+C.
"""

import argparse
import base64
import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, OpenAI

AUDIO_DIR = Path("data/audio")  # all audio files live here
TMP_DIR = AUDIO_DIR / "tmp"
OUTPUT_DIR = Path("data/outputs")  # generated transcripts
CHUNK_SECONDS = 25  # below the model's 30 s limit
TRANSCRIBE_PROMPT = "Transcribe this audio exactly."
FIRST_QUESTION_PROMPT = "What would you like to know about this audio? "

from services.config import require_env, get_client, SAMPLING_PARAMETERS


def resolve_audio_path(name: str) -> Path:
    """Map a bare file name to data/audio/<name>, rejecting anything with a folder part."""
    if name in {"", ".", ".."} or ".." in name or "/" in name or "\\" in name:
        raise ValueError(
            f"Pass only the file name (e.g. pigs.m4a), not a path: {name!r}. "
            f"Audio files must be inside {AUDIO_DIR}/."
        )
    path = AUDIO_DIR / name
    if not path.resolve().is_relative_to(AUDIO_DIR.resolve()):
        raise ValueError("Audio must be inside data/audio/.")
    if not path.is_file():
        raise ValueError(f"{name} not found in {AUDIO_DIR}/")
    return path


def run_ffmpeg(*args: str) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"ffmpeg failed:\n{err.stderr.strip()}")


def cli_split_into_chunks(audio_path: Path, work_dir: Path) -> list[Path]:
    """Convert to 16 kHz mono 16-bit WAV, then split into CHUNK_SECONDS-long chunks."""
    if work_dir.exists():
        shutil.rmtree(work_dir)  # leftovers from an earlier crashed run
    work_dir.mkdir(parents=True)

    return split_into_chunks(convert_to_wav(audio_path, work_dir), work_dir)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it (macOS: brew install ffmpeg)")


def convert_to_wav(src: Path, workdir: Path) -> Path:
    require_ffmpeg()
    workdir.mkdir(parents=True, exist_ok=True)
    wav = workdir / "full.wav"
    run_ffmpeg("-i", str(src), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav))
    return wav


def split_into_chunks(wav: Path, workdir: Path, seconds: int = CHUNK_SECONDS) -> list[Path]:
    require_ffmpeg()
    if seconds <= 0:
        raise ValueError("Chunk duration must be positive.")
    workdir.mkdir(parents=True, exist_ok=True)
    run_ffmpeg("-i", str(wav), "-f", "segment", "-segment_time", str(seconds),
               "-c", "copy", str(workdir / "chunk_%03d.wav"))
    return sorted(workdir.glob("chunk_*.wav"))


def iter_transcribe(src: Path):
    from tempfile import TemporaryDirectory
    # The context manager cleans up on exhaustion, failure and generator.close().
    with TemporaryDirectory(prefix="openweights-audio-") as directory:
        workdir = Path(directory)
        chunks = split_into_chunks(convert_to_wav(src, workdir), workdir)
        if not chunks:
            raise RuntimeError("ffmpeg produced no audio chunks.")
        texts = []
        for index, chunk in enumerate(chunks, 1):
            text = transcribe_chunk(chunk)
            if text:
                texts.append(text)
            yield {"type": "progress", "index": index, "total": len(chunks), "text": text}
        yield {"type": "done", "transcript": " ".join(texts)}


def build_transcript_context(transcript: str) -> dict:
    return {"role": "user", "content": "Here is the full transcript of an audio recording:\n\n" + transcript}


def transcribe_chunk(chunk: Path) -> str:
    audio_b64 = base64.b64encode(chunk.read_bytes()).decode("ascii")
    response = get_client().chat.completions.create(
        model=require_env("MODEL"),
        **SAMPLING_PARAMETERS,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIBE_PROMPT},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


def transcribe_all(client: OpenAI, model: str, chunks: list[Path], out_file: Path) -> str:
    """Transcribe chunks in order, saving each transcript to out_file as it arrives."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transcripts = []
    with out_file.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            print(f"Transcribing chunk {i}/{len(chunks)} ({chunk.name})...", flush=True)
            try:
                text = transcribe_chunk(chunk)
            except APIError as err:
                raise SystemExit(
                    f"The server rejected the audio request on {chunk.name}: {err}\n"
                    "Check that BASE_URL and MODEL point to a server that accepts "
                    "`input_audio` with an audio-capable model (e.g. llama-server). "
                    f"Temp files kept in {chunk.parent}/ for inspection."
                )
            print(f"  {text}\n")
            f.write(text + "\n")
            f.flush()
            if text:
                transcripts.append(text)
    return " ".join(transcripts)


def cleanup(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        TMP_DIR.rmdir()  # only removes it if now empty
    except OSError:
        pass


from services.chat import stream_reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "audio",
        help=f"name of the audio file inside {AUDIO_DIR} (e.g. pigs.m4a)",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found. Install it (brew install ffmpeg on macOS, "
            "apt install ffmpeg on Linux) and check with: ffmpeg -version"
        )

    audio_path = resolve_audio_path(args.audio)

    base_url = require_env("BASE_URL")
    model = require_env("MODEL")
    api_key = require_env("API_KEY")
    client = get_client()

    work_dir = TMP_DIR / audio_path.stem
    out_file = OUTPUT_DIR / f"{audio_path.stem}.txt"

    print(f"Audio: {audio_path} | Model: {model} at {base_url}\n")
    chunks = cli_split_into_chunks(audio_path, work_dir)
    if not chunks:
        raise SystemExit(f"ffmpeg produced no audio chunks from {audio_path}")

    transcript = transcribe_all(client, model, chunks, out_file)
    cleanup(work_dir)
    if not transcript:
        raise SystemExit("The model returned an empty transcript.")
    print(f"Transcript saved to {out_file}\n")

    messages = []
    first_question = True
    while True:
        try:
            user_input = input(FIRST_QUESTION_PROMPT if first_question else "You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        if first_question:
            content = (
                "Here is the full transcript of an audio recording:\n\n"
                f"{transcript}\n\n"
                f"Question: {user_input}"
            )
            first_question = False
        else:
            content = user_input

        messages.append({"role": "user", "content": content})
        reply = stream_reply(client, model, messages)
        messages.append({"role": "assistant", "content": reply})

