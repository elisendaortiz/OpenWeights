from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from services import audio, agent, image
from ui_pages import common


@pytest.fixture(autouse=True)
def connected(monkeypatch):
    monkeypatch.setattr(common, 'server_status', lambda: (True, 'ok'))
    monkeypatch.setattr(common, 'stream_chat', lambda messages: (token for token in ['Hello', ' world']))


def page(name):
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / f'ui_pages/{name}_page.py', default_timeout=30).run()
    assert not app.exception
    return app


def test_navigation_and_chat():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == 'Chatbot'
    app.chat_input[0].set_value('Hi').run()
    app.chat_input[0].set_value('Again').run()
    assert len(app.session_state.chat_messages) == 4
    assert app.session_state.chat_messages[-1]['content'] == 'Hello world'
    app.session_state['audio_transcript'] = 'keep me'
    app.button[0].click().run()
    assert app.session_state.chat_messages == []
    assert app.session_state.audio_transcript == 'keep me'


@pytest.mark.parametrize('name', ['chat', 'image', 'audio', 'agent'])
def test_offline(name, monkeypatch):
    monkeypatch.setattr(common, 'server_status', lambda: (False, 'Server unavailable'))
    app = page(name)
    assert app.error
    assert not app.chat_input


class Upload(BytesIO):
    def __init__(self, data, name, file_id):
        super().__init__(data)
        self.name = name
        self.file_id = file_id


def test_image_followup_and_change(monkeypatch):
    raw = BytesIO()
    Image.new('RGB', (10, 10)).save(raw, 'PNG')
    upload = Upload(raw.getvalue(), 'photo.png', 'one')
    with patch('streamlit.file_uploader', return_value=upload) as uploader:
        app = page('image')
        app.chat_input[0].set_value('Describe').run()
        app.chat_input[0].set_value('What color?').run()
        messages = app.session_state.image_messages
        assert isinstance(messages[0]['content'], list)
        assert isinstance(messages[2]['content'], str)
        assert len(messages) == 4
        uploader.return_value = Upload(raw.getvalue(), 'photo.png', 'two')
        app.run()
        assert app.session_state.image_messages == []
        uploader.return_value = None
        app.run()
        assert app.session_state.image_data_url is None
        assert app.chat_input[0].disabled
    assert not app.exception


def test_invalid_image():
    with patch('streamlit.file_uploader', return_value=Upload(b'invalid', 'bad.jpg', 'bad')):
        app = page('image')
        assert app.error
        assert app.chat_input[0].disabled


def test_audio_upload_cleanup_and_followups(monkeypatch):
    paths = []
    contexts = []
    def transcribe(path):
        assert path.name == 'input.m4a'
        assert path.exists()
        paths.append(path)
        yield {'type': 'progress', 'index': 1, 'total': 1, 'text': 'A recording'}
        yield {'type': 'done', 'transcript': 'A recording'}
    def stream(messages):
        contexts.append(messages)
        yield 'Answer'
    monkeypatch.setattr(audio, 'require_ffmpeg', lambda: None)
    monkeypatch.setattr(audio, 'iter_transcribe', transcribe)
    monkeypatch.setattr(common, 'stream_chat', stream)
    with patch('streamlit.file_uploader', return_value=Upload(b'audio', 'personal.m4a', 'audio')) as uploader:
        app = page('audio')
        assert app.chat_input[0].disabled
        app.button[1].click().run()
        assert app.session_state.audio_transcript == 'A recording'
        assert not paths[0].parent.exists()
        app.chat_input[0].set_value('Summarize').run()
        app.chat_input[0].set_value('Tell me more').run()
        assert len(app.session_state.audio_messages) == 4
        assert contexts[-1][0] == audio.build_transcript_context('A recording')
        uploader.return_value = Upload(b'new', 'new.m4a', 'new')
        app.run()
        assert app.session_state.audio_transcript == ''
        assert app.session_state.audio_messages == []
    assert not app.exception


def test_agent_tool_events(monkeypatch):
    events = [
        {'type': 'tool_call', 'name': 'get_current_time', 'arguments': {'timezone': 'Asia/Tokyo'}},
        {'type': 'tool_result', 'name': 'get_current_time', 'content': '12:00'},
        {'type': 'final', 'content': 'It is noon.'},
    ]
    monkeypatch.setattr(agent, 'run_agent_sync', lambda messages: (event for event in events))
    app = page('agent')
    app.chat_input[0].set_value('Tokyo?').run()
    assert not app.exception
    assert app.expander[0].label == 'Tool call: get_current_time'
    assert app.session_state.agent_messages[-1]['content'] == 'It is noon.'
    assert all(m['role'] in {'user', 'assistant'} for m in app.session_state.agent_messages)


def test_empty_and_failed_chat(monkeypatch):
    monkeypatch.setattr(common, 'stream_chat', lambda messages: (token for token in []))
    app = page('chat')
    app.chat_input[0].set_value('Hello').run()
    assert app.info
    assert app.session_state.chat_messages == []
    def fail(messages):
        raise RuntimeError('Test failure')
    monkeypatch.setattr(common, 'stream_chat', fail)
    app.chat_input[0].set_value('Retry').run()
    assert app.error
    assert not app.exception
    assert app.session_state.chat_messages == []
