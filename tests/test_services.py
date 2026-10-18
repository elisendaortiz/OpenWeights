import base64
import io
import shutil
import wave
from pathlib import Path

import pytest
from PIL import Image
from services import audio, agent
from services.chat import message_text
from services.image import prepare_image, load_image_from_data_dir


def test_image_resize_and_invalid():
    source = io.BytesIO()
    Image.new('RGBA', (1800, 1200)).save(source, 'PNG')
    url = prepare_image(source.getvalue())
    assert url.startswith('data:image/jpeg;base64,')
    with Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))) as img:
        assert max(img.size) <= 1024
        assert img.mode == 'RGB'
    with pytest.raises(ValueError):
        prepare_image(b'not an image')


@pytest.mark.parametrize('name', ['../x.jpg', 'a/b', 'a\\b', '..', '', 'a..b'])
def test_paths_rejected(name):
    for loader in (load_image_from_data_dir, audio.resolve_audio_path):
        with pytest.raises(ValueError):
            loader(name)


def test_message_text():
    assert message_text({'content': 'hello'}) == 'hello'
    assert message_text({'content': [{'type': 'text', 'text': 'hi'}, {'type': 'image_url', 'image_url': {'url': 'secret'}}, {'type': 'text', 'text': 'there'}]}) == 'hi\nthere'
    assert message_text({'content': None}) == ''


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg missing')
def test_chunk_order(tmp_path):
    source = tmp_path / 'silence.wav'
    with wave.open(str(source), 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\0\0' * 16000 * 60)
    work = tmp_path / 'chunks'
    chunks = audio.split_into_chunks(audio.convert_to_wav(source, work), work)
    assert [p.name for p in chunks] == ['chunk_000.wav', 'chunk_001.wav', 'chunk_002.wav']
    for chunk in chunks:
        with wave.open(str(chunk)) as wav:
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2


@pytest.mark.parametrize('early,fail', [(True, False), (False, False), (False, True)])
def test_transcription_cleanup(monkeypatch, early, fail):
    paths = []
    def convert(src, work):
        paths.append(work)
        return work / 'full.wav'
    monkeypatch.setattr(audio, 'convert_to_wav', convert)
    monkeypatch.setattr(audio, 'split_into_chunks', lambda wav, work: [work/'a', work/'b'])
    def transcribe(path):
        if fail:
            raise RuntimeError('failed')
        return path.name
    monkeypatch.setattr(audio, 'transcribe_chunk', transcribe)
    gen = audio.iter_transcribe(Path('input'))
    if fail:
        with pytest.raises(RuntimeError):
            list(gen)
    elif early:
        assert next(gen)['index'] == 1
        gen.close()
    else:
        assert list(gen)[-1] == {'type': 'done', 'transcript': 'a b'}
    assert not paths[0].exists()


def test_agent_events_and_failure(monkeypatch):
    async def run(messages, max_iterations, emit):
        emit({'type': 'final', 'content': 'hello'})
    monkeypatch.setattr(agent, '_run_agent', run)
    assert list(agent.run_agent_sync([])) == [{'type': 'final', 'content': 'hello'}]
    async def fail(*args):
        raise RuntimeError('test failure')
    monkeypatch.setattr(agent, '_run_agent', fail)
    assert list(agent.run_agent_sync([]))[0]['type'] == 'error'


def test_agent_close_cancels_owner(monkeypatch):
    import asyncio
    from threading import Event
    closed = Event()
    async def run(messages, max_iterations, emit):
        try:
            emit({'type': 'tool_call', 'name': 'test', 'arguments': {}})
            await asyncio.sleep(30)
        finally:
            closed.set()
    monkeypatch.setattr(agent, '_run_agent', run)
    gen = agent.run_agent_sync([])
    next(gen)
    gen.close()
    assert closed.wait(1)
