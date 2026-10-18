"""Run with: uv run uvicorn api.main:app --host 127.0.0.1 --port 8000."""
from collections.abc import Callable, Iterator
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated

import anyio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, Field, field_validator

from services.agent import run_agent_sync
from services.audio import build_transcript_context, iter_transcribe, require_ffmpeg
from services.chat import stream_chat
from services.config import MODEL, check_server
from services.image import build_image_message, prepare_image

app = FastAPI(title='OpenWeights Playground', docs_url=None, redoc_url=None)
IMAGE_LIMIT = 10 * 1024 * 1024
AUDIO_LIMIT = 50 * 1024 * 1024
IMAGE_TYPES = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp'}
AUDIO_EXTENSIONS = {'.m4a', '.mp3', '.wav', '.ogg', '.aac'}
model_lock = Lock()


class Conversation(BaseModel):
    messages: list[dict] = Field(min_length=1)

    @field_validator('messages')
    @classmethod
    def valid_messages(cls, messages):
        for message in messages:
            if message.get('role') not in {'system', 'user', 'assistant'}:
                raise ValueError('Messages must have a system, user or assistant role.')
            content = message.get('content')
            if not isinstance(content, (str, list)):
                raise ValueError('Message content must be text or content parts.')
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        raise ValueError('Each content part must be an object.')
                    if part.get('type') == 'text' and isinstance(part.get('text'), str):
                        continue
                    if part.get('type') == 'image_url':
                        url = part.get('image_url', {}).get('url') if isinstance(part.get('image_url'), dict) else None
                        if isinstance(url, str) and url.startswith('data:image/'):
                            continue
                    raise ValueError('Use text parts or prepared image data URLs.')
        return messages


class AgentConversation(Conversation):
    @field_validator('messages')
    @classmethod
    def plain_messages(cls, messages):
        if any(m['role'] not in {'user', 'assistant'} or not isinstance(m['content'], str) for m in messages):
            raise ValueError('Agent history must contain plain user and assistant text only.')
        return messages


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Pydantic's default response includes input values, potentially base64 data.
    return JSONResponse(status_code=422, content={'detail': 'Invalid request. Provide the required fields and valid conversation messages.'})


def error_message(exc: Exception) -> str:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return 'Cannot reach the model server. Is Ollama running? Check BASE_URL and try again.'
    if isinstance(exc, APIStatusError):
        return f'The model server rejected the request (HTTP {exc.status_code}). Check MODEL and support for this input type.'
    if isinstance(exc, (ValueError, RuntimeError)):
        return str(exc)
    return f'Request failed ({type(exc).__name__}). Check your input and try again.'


class EventStream(StreamingResponse):
    """Explicitly close sync generators and uploads even on client disconnect."""
    def __init__(self, events: Iterator[dict], cleanup: Callable | None = None):
        self.events = events
        self.cleanup = cleanup
        self.encoded = self.encode()
        super().__init__(self.encoded, media_type='text/event-stream', headers={
            'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
        })

    def encode(self):
        try:
            for event in self.events:
                yield 'data: ' + json.dumps(event, ensure_ascii=False) + '\n\n'
                if event.get('type') == 'error':
                    break
        except Exception as exc:
            yield 'data: ' + json.dumps({'type': 'error', 'message': error_message(exc)}) + '\n\n'
        finally:
            close = getattr(self.events, 'close', None)
            if close:
                close()

    async def __call__(self, scope, receive, send):
        try:
            # Always listen for disconnects, including ASGI >=2.4. Sync service
            # calls stay in the threadpool; an in-flight next() finishes before
            # its generator is closed, avoiding cross-thread generator teardown.
            async with anyio.create_task_group() as group:
                async def stream():
                    await self.stream_response(send)
                    group.cancel_scope.cancel()
                group.start_soon(stream)
                await self.listen_for_disconnect(receive)
                group.cancel_scope.cancel()
        finally:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(self.encoded.close)
                if self.cleanup:
                    await anyio.to_thread.run_sync(self.cleanup)


def sse(events: Iterator[dict], cleanup: Callable | None = None) -> StreamingResponse:
    return EventStream(events, cleanup)


def one_request(events: Iterator[dict]) -> Iterator[dict]:
    """Avoid competing model requests on small-memory machines."""
    if not model_lock.acquire(blocking=False):
        events.close()
        yield {'type': 'error', 'message': 'The model is still working on another request. Please try again shortly.'}
        return
    try:
        with closing(events):
            yield from events
    finally:
        model_lock.release()


def save_upload(file: UploadFile, path: Path, limit: int):
    size = 0
    try:
        with path.open('wb') as target:
            while chunk := file.file.read(64 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f'File exceeds the {limit // (1024 * 1024)} MB limit.')
                target.write(chunk)
        if not size:
            raise HTTPException(400, 'The uploaded file is empty.')
    finally:
        file.file.close()


@app.get('/api/health')
def health():
    ok, detail = check_server()
    return {'ok': ok, 'detail': detail, 'model': MODEL}


@app.post('/api/chat')
def chat(body: Conversation):
    def events():
        with closing(stream_chat(body.messages)) as tokens:
            for text in tokens:
                yield {'type': 'delta', 'text': text}
        yield {'type': 'done'}
    return sse(one_request(events()))


@app.post('/api/image/prepare')
def image_prepare(file: Annotated[UploadFile, File()], prompt: Annotated[str, Form()]):
    try:
        if file.content_type not in IMAGE_TYPES:
            raise HTTPException(415, 'Upload a JPEG, PNG or WebP image.')
        with TemporaryDirectory(prefix='openweights-image-') as directory:
            path = Path(directory) / ('input' + IMAGE_TYPES[file.content_type])
            save_upload(file, path, IMAGE_LIMIT)
            try:
                data_url = prepare_image(path.read_bytes())
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return {'message': build_image_message(data_url, prompt)}
    finally:
        file.file.close()


@app.post('/api/audio/transcribe')
def audio_transcribe(file: Annotated[UploadFile, File()]):
    directory = None
    try:
        extension = Path(file.filename or '').suffix.lower()
        if extension not in AUDIO_EXTENSIONS:
            raise HTTPException(415, 'Upload an m4a, mp3, wav, ogg or aac recording.')
        try:
            require_ffmpeg()
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        directory = TemporaryDirectory(prefix='openweights-upload-')
        source = Path(directory.name) / ('input' + extension)
        save_upload(file, source, AUDIO_LIMIT)
    except BaseException:
        if directory:
            directory.cleanup()
        raise
    finally:
        file.file.close()

    def events():
        try:
            with closing(iter_transcribe(source)) as transcription:
                for event in transcription:
                    if event['type'] == 'done':
                        if not event['transcript']:
                            raise ValueError('The model returned an empty transcript. Try another recording.')
                        event = {**event, 'context_message': build_transcript_context(event['transcript'])}
                    yield event
        finally:
            directory.cleanup()
    # The callback also handles a response that disconnects before iteration.
    return sse(one_request(events()), cleanup=directory.cleanup)


@app.post('/api/agent')
def agent(body: AgentConversation):
    return sse(one_request(run_agent_sync(body.messages)))


app.mount('/', StaticFiles(directory=Path(__file__).resolve().parents[1] / 'web', html=True), name='web')
