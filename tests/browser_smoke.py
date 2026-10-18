"""Optional real-browser checks. No model is used.

Run: uv run --with playwright python tests/browser_smoke.py
Uses an installed Google Chrome; no frontend tooling or runtime dependency.
"""
from io import BytesIO
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SERVER = '''
import time
import uvicorn
from api import main
online = True
main.check_server = lambda: (online, 'ok' if online else 'offline')
main.require_ffmpeg = lambda: None
@main.app.post('/test/offline')
def offline():
    global online
    online = False
    return {'ok': True}
# Test-only routes must precede the static mount.
main.app.router.routes.insert(0, main.app.router.routes.pop())
def chat(messages):
    last = messages[-1]['content']
    if last == 'fail':
        raise RuntimeError('A friendly test error')
    count = sum(m['role'] == 'user' for m in messages)
    images = sum(isinstance(m['content'], list) for m in messages)
    if last == 'slow':
        for i in range(80):
            yield 'partial '
            time.sleep(.05)
    else:
        for text in [f'Turn {count}; images {images}. ', '<img src=x onerror=alert(1)> ', 'Hello 世界.']:
            yield text
            time.sleep(.08)
main.stream_chat = chat
def transcribe(path):
    assert path.name == 'input.m4a'
    for i in [1, 2]:
        time.sleep(.08)
        yield {'type': 'progress', 'index': i, 'total': 2, 'text': f'Part {i}.'}
    yield {'type': 'done', 'transcript': 'Part 1. Part 2.'}
main.iter_transcribe = transcribe
def agent(messages):
    yield {'type': 'tool_call', 'name': 'get_current_time', 'arguments': {'timezone': 'Asia/Tokyo'}}
    time.sleep(.08)
    yield {'type': 'tool_result', 'name': 'get_current_time', 'content': '12:00'}
    yield {'type': 'final', 'content': 'It is noon in Tokyo.'}
main.run_agent_sync = agent
uvicorn.run(main.app, host='127.0.0.1', port=PORT, log_level='error')
'''


def main():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen([sys.executable, '-c', SERVER.replace('port=PORT', f'port={port}')], cwd=ROOT)
    base = f'http://127.0.0.1:{port}'
    try:
        for _ in range(80):
            try:
                urllib.request.urlopen(base + '/api/health', timeout=1).close()
                break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError('Test server did not start')
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='chrome', headless=True)
            context = browser.new_context(viewport={'width': 1440, 'height': 1000}, color_scheme='light')
            page = context.new_page()
            errors = []
            external = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: external.append(request.url) if not request.url.startswith(base) and not request.url.startswith(('blob:', 'data:')) else None)
            page.goto(base)
            page.get_by_text('Server connected', exact=True).wait_for()
            shots = ROOT / '.cache' / 'browser'
            shots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(shots / 'desktop-light.png'))
            def send(prompt):
                page.locator('#prompt').fill(prompt)
                page.locator('#prompt').press('Enter')
                page.wait_for_function("document.querySelector('#send-label').textContent === 'Send'")
            def panel(name):
                page.locator(f'[data-panel="{name}"]').click()
            send('hello')
            assert 'Turn 1' in page.locator('#messages').inner_text()
            assert page.locator('#messages img').count() == 0  # output is literal text, never HTML
            send('follow up')
            assert 'Turn 2' in page.locator('#messages').inner_text()
            page.locator('#prompt').fill('slow')
            page.locator('#prompt').press('Enter')
            page.get_by_text('partial', exact=False).last.wait_for()
            page.get_by_role('button', name='Stop response').click()
            page.wait_for_function("document.querySelector('#send-label').textContent === 'Send'")
            assert 'Stopped · partial answer' in page.locator('#messages').inner_text()
            panel('image')
            assert page.locator('#prompt').is_disabled()
            image = BytesIO()
            Image.new('RGB', (100, 70), '#326c53').save(image, 'PNG')
            payload = {'name': 'photo.png', 'mimeType': 'image/png', 'buffer': image.getvalue()}
            page.locator('#file-input').set_input_files(payload)
            send('Describe')
            send('Color?')
            assert 'Turn 2; images 1' in page.locator('#messages').inner_text()
            assert 'base64' not in page.locator('#messages').inner_text()
            page.locator('#file-input').set_input_files({**payload, 'name': 'replacement.png'})
            assert page.locator('.message-row').count() == 0
            page.get_by_role('button', name='Remove attachment').click()
            assert page.locator('#prompt').is_disabled()
            # Exercise the drag-and-drop handler with a real DataTransfer/File.
            page.locator('#drop-zone').evaluate('''(zone, bytes) => {
                const transfer = new DataTransfer();
                transfer.items.add(new File([new Uint8Array(bytes)], 'drop.png', {type:'image/png'}));
                zone.dispatchEvent(new DragEvent('drop', {dataTransfer:transfer, bubbles:true}));
            }''', list(image.getvalue()))
            assert page.locator('#prompt').is_enabled()
            panel('audio')
            page.locator('#file-input').set_input_files({'name': 'clip.m4a', 'mimeType': 'audio/mp4', 'buffer': b'audio'})
            page.get_by_role('button', name='Transcribe', exact=True).click()
            page.wait_for_function("!document.querySelector('#prompt').disabled")
            assert 'Chunk 2 of 2' in page.locator('#transcription-progress').inner_text()
            page.locator('#transcript summary').click()
            assert page.locator('#transcript-text').inner_text() == 'Part 1. Part 2.'
            send('Summarize')
            send('More?')
            assert 'Turn 3' in page.locator('#messages').inner_text()  # transcript context + two questions
            panel('agent')
            send('Tokyo?')
            assert page.locator('.tool-chip').inner_text().find('Asia/Tokyo') >= 0
            assert 'It is noon in Tokyo.' in page.locator('#messages').inner_text()
            send('And now?')
            assert page.locator('.tool-chip').count() == 2
            panel('chat')
            assert 'Stopped · partial answer' in page.locator('#messages').inner_text()
            page.get_by_role('button', name='New chat').click()
            assert page.locator('.message-row').count() == 0
            send('fail')
            assert 'A friendly test error' in page.locator('.error').inner_text()
            panel('agent')
            assert page.locator('.tool-chip').count() == 2
            page.screenshot(path=str(shots / 'agent-light.png'))
            page.emulate_media(color_scheme='dark')
            page.screenshot(path=str(shots / 'desktop-dark.png'))
            page.set_viewport_size({'width': 390, 'height': 844})
            page.screenshot(path=str(shots / 'mobile-dark.png'))
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert page.locator('#send').is_visible()
            page.emulate_media(color_scheme='light')
            panel('chat')
            page.screenshot(path=str(shots / 'mobile-light.png'))
            page.request.post(base + '/test/offline')
            page.reload()
            page.locator('#offline-banner').wait_for(state='visible')
            assert page.locator('#prompt').is_enabled()
            assert not errors, errors
            assert not external, external
            print('Browser checks passed: chat, Stop, image, drag/drop, audio, agent, panel state, errors, offline banner, dark/light and mobile layout.')
            print('Screenshots:', shots)
            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == '__main__':
    main()
