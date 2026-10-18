import asyncio
from io import BytesIO
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api import main


@pytest.fixture
def client():
    with TestClient(main.app) as client:
        yield client


def events(response):
    assert response.headers['content-type'].startswith('text/event-stream')
    assert response.headers['cache-control'] == 'no-cache'
    assert response.headers['x-accel-buffering'] == 'no'
    return [json.loads(block[6:]) for block in response.text.strip().split('\n\n')]


def test_health_and_static(client, monkeypatch):
    monkeypatch.setattr(main, 'check_server', lambda: (False, 'offline'))
    assert client.get('/api/health').json() == {'ok': False, 'detail': 'offline', 'model': main.MODEL}
    page = client.get('/')
    assert page.status_code == 200
    assert 'OpenWeights Playground' in page.text
    assert all(label in page.text for label in ['Chatbot', 'Images', 'Audio', 'Agent'])
    assert client.get('/app.js').status_code == 200
    assert client.get('/style.css').status_code == 200
    assert client.get('/.env').status_code == 404


def test_chat_stream_and_error(client, monkeypatch):
    closed = []
    received = []
    def stream(messages):
        received.extend(messages)
        try:
            yield 'Hello '
            yield '世界\nagain'
        finally:
            closed.append(True)
    monkeypatch.setattr(main, 'stream_chat', stream)
    messages = [{'role': 'user', 'content': 'hello'}]
    assert events(client.post('/api/chat', json={'messages': messages})) == [
        {'type': 'delta', 'text': 'Hello '}, {'type': 'delta', 'text': '世界\nagain'}, {'type': 'done'}]
    assert received == messages
    assert closed == [True]
    def fail(messages):
        yield 'partial'
        raise RuntimeError('test failure')
    monkeypatch.setattr(main, 'stream_chat', fail)
    result = events(client.post('/api/chat', json={'messages': messages}))
    assert result[-1] == {'type': 'error', 'message': 'test failure'}
    assert not main.model_lock.locked()


def png():
    output = BytesIO()
    Image.new('RGB', (10, 10)).save(output, 'PNG')
    return output.getvalue()


def test_image_prepare(client):
    response = client.post('/api/image/prepare', files={'file': ('../../private.png', png(), 'image/png')}, data={'prompt': 'Describe'})
    assert response.status_code == 200
    message = response.json()['message']
    assert message['role'] == 'user'
    assert message['content'][0] == {'type': 'text', 'text': 'Describe'}
    assert message['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert client.post('/api/image/prepare', files={'file': ('bad.txt', b'text', 'text/plain')}, data={'prompt': 'Hi'}).status_code == 415
    assert client.post('/api/image/prepare', files={'file': ('bad.png', b'text', 'image/png')}, data={'prompt': 'Hi'}).status_code == 400


def test_image_size_limit(client, monkeypatch):
    monkeypatch.setattr(main, 'IMAGE_LIMIT', 3)
    assert client.post('/api/image/prepare', files={'file': ('big.png', b'1234', 'image/png')}, data={'prompt': 'Hi'}).status_code == 413


def test_audio_stream_and_cleanup(client, monkeypatch):
    paths = []
    monkeypatch.setattr(main, 'require_ffmpeg', lambda: None)
    def transcribe(path):
        paths.append(path)
        assert path.name == 'input.m4a'
        assert path.read_bytes() == b'audio'
        yield {'type': 'progress', 'index': 1, 'total': 1, 'text': 'A transcript'}
        yield {'type': 'done', 'transcript': 'A transcript'}
    monkeypatch.setattr(main, 'iter_transcribe', transcribe)
    response = client.post('/api/audio/transcribe', files={'file': ('../../secret.M4A', b'audio', 'audio/mp4')})
    result = events(response)
    assert result[0]['type'] == 'progress'
    assert result[1] == {'type': 'done', 'transcript': 'A transcript', 'context_message': main.build_transcript_context('A transcript')}
    assert not paths[0].parent.exists()


def test_audio_validation_and_cleanup(client, monkeypatch):
    directories = []
    real_temp = main.TemporaryDirectory
    def temporary(*args, **kwargs):
        directory = real_temp(*args, **kwargs)
        directories.append(Path(directory.name))
        return directory
    monkeypatch.setattr(main, 'TemporaryDirectory', temporary)
    monkeypatch.setattr(main, 'require_ffmpeg', lambda: None)
    monkeypatch.setattr(main, 'AUDIO_LIMIT', 3)
    response = client.post('/api/audio/transcribe', files={'file': ('a.m4a', b'1234')})
    assert response.status_code == 413
    assert directories and all(not path.exists() for path in directories)
    assert client.post('/api/audio/transcribe', files={'file': ('a.exe', b'x')}).status_code == 415
    assert client.post('/api/audio/transcribe', files={'file': ('a.wav', b'')}).status_code == 400
    def missing():
        raise RuntimeError('ffmpeg not found. Install it (macOS: brew install ffmpeg)')
    monkeypatch.setattr(main, 'require_ffmpeg', missing)
    response = client.post('/api/audio/transcribe', files={'file': ('a.wav', b'1')})
    assert response.status_code == 500
    assert 'brew install ffmpeg' in response.json()['detail']


@pytest.mark.parametrize('empty', [False, True])
def test_audio_failed_stream_cleanup(client, monkeypatch, empty):
    paths = []
    monkeypatch.setattr(main, 'require_ffmpeg', lambda: None)
    def transcribe(path):
        paths.append(path)
        if empty:
            yield {'type': 'done', 'transcript': ''}
        else:
            raise RuntimeError('invalid audio')
    monkeypatch.setattr(main, 'iter_transcribe', transcribe)
    result = events(client.post('/api/audio/transcribe', files={'file': ('a.wav', b'broken')}))
    assert result[-1]['type'] == 'error'
    assert not paths[0].parent.exists()


def test_agent(client, monkeypatch):
    expected = [
        {'type': 'tool_call', 'name': 'get_current_time', 'arguments': {'timezone': 'Asia/Tokyo'}},
        {'type': 'tool_result', 'name': 'get_current_time', 'content': '12:00'},
        {'type': 'final', 'content': 'Noon'},
    ]
    monkeypatch.setattr(main, 'run_agent_sync', lambda messages: (event for event in expected))
    assert events(client.post('/api/agent', json={'messages': [{'role': 'user', 'content': 'Tokyo?'}]})) == expected
    assert client.post('/api/agent', json={'messages': [{'role': 'system', 'content': 'No'}]}).status_code == 422
    assert client.post('/api/agent', json={'messages': [{'role': 'user', 'content': []}]}).status_code == 422


@pytest.mark.parametrize('messages', [[], [{}], [{'role': 'user', 'content': 5}], [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'https://external/image'}}]}]])
def test_invalid_messages(client, messages):
    assert client.post('/api/chat', json={'messages': messages}).status_code == 422


def test_busy(client, monkeypatch):
    calls = []
    def stream(messages):
        calls.append(True)
        yield 'should not run'
    monkeypatch.setattr(main, 'stream_chat', stream)
    with main.model_lock:
        result = events(client.post('/api/chat', json={'messages': [{'role': 'user', 'content': 'Hi'}]}))
    assert result[0]['type'] == 'error'
    assert not calls


def test_disconnect_closes_generator_and_releases_lock():
    closed = []
    cleaned = []
    def source():
        try:
            while True:
                yield {'type': 'delta', 'text': 'hello'}
        finally:
            closed.append(True)
    async def run():
        disconnected = asyncio.Event()
        response = main.sse(main.one_request(source()), cleanup=lambda: cleaned.append(True))
        async def receive():
            await disconnected.wait()
            return {'type': 'http.disconnect'}
        async def send(message):
            if message['type'] == 'http.response.body':
                disconnected.set()
        await response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, send)
    asyncio.run(run())
    assert closed == [True]
    assert cleaned == [True]
    assert not main.model_lock.locked()
