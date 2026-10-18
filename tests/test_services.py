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


@pytest.mark.parametrize('tool_arguments,cap', [('{"timezone":"Asia/Tokyo"}', 5), ('bad json', 5), ('{}', 1)])
def test_agent_protocol(monkeypatch, tool_arguments, cap):
    import asyncio
    from contextlib import asynccontextmanager
    from types import SimpleNamespace as NS
    requests = []
    tool_calls = []
    call = NS(id='call-1', function=NS(name='get_current_time', arguments=tool_arguments))
    class Message:
        content = None
        def __init__(self, calls):
            self.tool_calls = calls
        def model_dump(self, **kwargs):
            return {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'call-1', 'type': 'function', 'function': {'name': 'get_current_time', 'arguments': tool_arguments}}]}
    async def create(**kwargs):
        requests.append(list(kwargs['messages']))
        message = Message([call] if len(requests) == 1 else [])
        if len(requests) > 1:
            message.content = 'It is noon.'
        return NS(choices=[NS(message=message)])
    class Session:
        async def initialize(self): pass
        async def list_tools(self):
            return NS(tools=[NS(name='get_current_time', description='Time', input_schema={'type': 'object'})])
        async def call_tool(self, name, arguments):
            tool_calls.append(arguments)
            return NS(content=[NS(type='text', text='12:00')], structured_content=None, is_error=False)
    @asynccontextmanager
    async def client(**kwargs):
        yield NS(chat=NS(completions=NS(create=create)))
    @asynccontextmanager
    async def stdio(params):
        yield (None, None)
    @asynccontextmanager
    async def session(*args):
        yield Session()
    monkeypatch.setattr(agent, 'AsyncOpenAI', client)
    monkeypatch.setattr(agent, 'stdio_client', stdio)
    monkeypatch.setattr(agent, 'ClientSession', session)
    monkeypatch.setattr(agent, 'require_env', lambda name: 'test')
    monkeypatch.setattr(agent.shutil, 'which', lambda name: '/test/uvx')
    messages = [{'role': 'user', 'content': 'Tokyo time?'}]
    events = []
    asyncio.run(agent._run_agent(messages, cap, events.append))
    assert len(messages) == 1  # tools stay internal to the run
    assert events[0]['type'] == 'tool_call'
    assert events[1]['type'] == 'tool_result'
    assert events[-1]['type'] == ('error' if cap == 1 else 'final')
    if cap > 1:
        assert requests[1][-1]['tool_call_id'] == 'call-1'
    if tool_arguments == 'bad json':
        assert not tool_calls
        assert 'Error:' in events[1]['content']
