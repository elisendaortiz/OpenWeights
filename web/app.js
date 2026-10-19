const $ = (selector) => document.querySelector(selector);
const panels = {
  chat: {title: 'Chatbot', description: 'A little curiosity goes a long way.', icon: 'chat', greeting: 'Where shall we begin?', hint: 'A question, a rough idea, a fresh perspective. Make some room for curiosity.', placeholder: 'Ask anything. Start a conversation.', caption: 'Keep exploring. Models can make mistakes.', suggestions: ['Explain something simply', 'Help me brainstorm', 'Write a short story']},
  image: {title: 'Images', description: 'A fresh perspective on what you see.', icon: 'image', greeting: 'There’s more to the picture.', hint: 'Bring a photo, a screenshot, or something that caught your eye. Let’s take a closer look.', placeholder: 'Ask something about the image, for example: Describe this image.', caption: 'Your image stays in this conversation. Models can miss details.'},
  audio: {title: 'Audio', description: 'Turn a recording into a conversation.', icon: 'audio', greeting: 'Listen. Transcribe. Explore.', hint: 'Add a recording and turn spoken words into a transcript you can ask questions about.', placeholder: 'Ask a question about the audio, for example: Summarize this audio.', caption: 'Answers use the transcript only; tone and non-speech sounds are not heard.'},
  agent: {title: 'Agent', description: 'A small assistant with a sense of time.', icon: 'agent', greeting: 'The right tool for the moment.', hint: 'Ask about time around the world. Watch your model call tools and work with the results.', placeholder: 'What time is it in Tokyo?', caption: 'Uses the MCP time server (uvx mcp-server-time). Try: What time is it in Tokyo?', suggestions: ['What time is it in Tokyo?', 'What time is it in London?']},
};
const freshState = () => ({messages: [], extra: {}, views: [], draft: ''});
const state = Object.fromEntries(Object.keys(panels).map((key) => [key, freshState()]));
let current = 'chat';
let operation = null;
let followBottom = true;
const messageList = $('#messages');

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.append(use);
  return svg;
}
function visibleText(message) {
  return typeof message.content === 'string' ? message.content : message.content.filter((part) => part.type === 'text').map((part) => part.text).join('\n');
}
function scrollBottom() {
  if (followBottom) messageList.scrollTop = messageList.scrollHeight;
}
messageList.addEventListener('scroll', () => {
  followBottom = messageList.scrollHeight - messageList.clientHeight - messageList.scrollTop < 65;
});
function renderHistory() {
  const position = messageList.scrollTop;
  const shouldFollow = followBottom;
  const panelState = state[current];
  messageList.replaceChildren();
  if (!panelState.views.length) {
    const empty = element('div', 'empty-state');
    const badge = element('div', 'empty-icon');
    badge.append(icon(panels[current].icon));
    empty.append(badge, element('h2', '', panels[current].greeting), element('p', '', panels[current].hint));
    if (panels[current].suggestions) {
      const suggestions = element('div', 'suggestions');
      for (const suggestion of panels[current].suggestions) {
        const button = element('button', 'suggestion', suggestion);
        button.disabled = Boolean(operation);
        button.addEventListener('click', () => {
          $('#prompt').value = suggestion;
          state[current].draft = suggestion;
          resizeComposer();
          $('#prompt').focus();
        });
        suggestions.append(button);
      }
      empty.append(suggestions);
    }
    messageList.append(empty);
  } else {
    if (current === 'image' && panelState.extra.preview) {
      const preview = element('img', 'history-image');
      preview.src = panelState.extra.preview;
      preview.alt = 'Attached image';
      messageList.append(preview);
    }
    for (const view of panelState.views) {
      if (view.kind === 'tool') {
        const details = element('details', 'tool-chip');
        details.open = Boolean(view.open);
        details.append(element('summary', '', `Tool call: ${view.name}`), element('pre', '', JSON.stringify(view.arguments, null, 2)));
        if (view.result !== undefined) details.append(element('pre', '', view.result));
        details.addEventListener('toggle', () => { view.open = details.open; });
        messageList.append(details);
        continue;
      }
      const row = element('div', `message-row ${view.kind}`);
      row.append(element('div', 'message-label', view.kind === 'user' ? 'You' : view.kind === 'error' ? 'Something went wrong' : 'OpenWeights'));
      const bubble = element('div', 'bubble');
      if (view.waiting && !view.text) {
        const typing = element('div', 'typing');
        typing.setAttribute('aria-label', 'Waiting for response');
        for (let i = 0; i < 3; i++) typing.append(element('span'));
        bubble.append(typing);
      } else bubble.textContent = view.text;
      view.node = bubble;
      row.append(bubble);
      if (view.note) row.append(element('div', 'note', view.note));
      messageList.append(row);
    }
  }
  followBottom = shouldFollow;
  if (shouldFollow) scrollBottom();
  else messageList.scrollTop = position;
}
function appendView(panel, view) {
  state[panel].views.push(view);
  if (current === panel) renderHistory();
  return view;
}
function updateAnswer(panel, view, text) {
  view.waiting = false;
  view.text = text;
  if (current === panel && view.node?.isConnected) {
    view.node.textContent = text;
    scrollBottom();
  }
}
function errorBubble(panel, message) {
  appendView(panel, {kind: 'error', text: message});
}
function prerequisites(panel) {
  return panel !== 'image' && panel !== 'audio' || panel === 'image' && Boolean(state.image.extra.file) || panel === 'audio' && Boolean(state.audio.extra.transcript);
}
function updateControls() {
  const busy = Boolean(operation);
  $('#prompt').disabled = busy || !prerequisites(current);
  $('#new-chat').disabled = busy;
  $('#choose-file').disabled = busy;
  $('#remove-file').disabled = busy;
  $('#file-input').disabled = busy;
  $('#transcribe').disabled = busy || !state.audio.extra.file;
  $('#send').disabled = !busy && !prerequisites(current);
  $('#send').classList.toggle('stopping', busy);
  $('#send').setAttribute('aria-label', busy ? 'Stop response' : 'Send message');
  $('#send-label').textContent = busy ? 'Stop' : 'Send';
  $('#operation-status').textContent = busy ? `${operation.kind === 'transcribe' ? 'Transcribing' : 'Working'} in ${panels[operation.panel].title}…` : '';
  document.querySelectorAll('.suggestion').forEach((button) => { button.disabled = busy; });
}
function renderAttachments() {
  const extra = state[current].extra;
  const isImage = current === 'image';
  const isAudio = current === 'audio';
  $('#attachment-area').hidden = !isImage && !isAudio;
  $('#audio-actions').hidden = !isAudio;
  $('#file-input').accept = isImage ? 'image/jpeg,image/png,image/webp' : '.m4a,.mp3,.wav,.ogg,.aac';
  $('#upload-title').textContent = extra.file ? extra.file.name : isImage ? 'Bring an image' : 'Add a recording';
  $('#upload-hint').textContent = extra.file ? `${(extra.file.size / 1024 / 1024).toFixed(2)} MB · Ready to explore` : isImage ? 'JPG, PNG or WebP · up to 10 MB · or drop here' : 'M4A, MP3, WAV, OGG, AAC · up to 50 MB · or drop here';
  $('#choose-file').textContent = extra.file ? 'Replace' : 'Choose file';
  $('#remove-file').hidden = !extra.file;
  $('#image-preview').hidden = !isImage || !extra.preview || state[current].views.length > 0;
  if (isImage && extra.preview) $('#image-preview').src = extra.preview;
  else $('#image-preview').removeAttribute('src');
  $('#transcript').hidden = !isAudio || !extra.transcript;
  $('#transcript-text').textContent = isAudio ? extra.transcript || '' : '';
  const progress = $('#transcription-progress');
  progress.hidden = !isAudio || !extra.progress?.length;
  progress.textContent = isAudio ? (extra.progress || []).map((event) => `Chunk ${event.index} of ${event.total}\n${event.text}`).join('\n\n') : '';
  if (isAudio) progress.scrollTop = progress.scrollHeight;
}
function renderPanel() {
  const panel = panels[current];
  $('#panel-title').textContent = panel.title;
  $('#panel-description').textContent = panel.description;
  $('#panel-caption').textContent = panel.caption;
  $('#prompt').placeholder = panel.placeholder;
  $('#prompt').value = state[current].draft;
  document.querySelectorAll('[data-panel]').forEach((button) => {
    button.classList.toggle('active', button.dataset.panel === current);
    if (button.dataset.panel === current) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  followBottom = true;
  renderHistory();
  renderAttachments();
  updateControls();
  resizeComposer();
}
function resetPanel(panel) {
  if (state[panel].extra.preview) URL.revokeObjectURL(state[panel].extra.preview);
  state[panel] = freshState();
  $('#file-input').value = '';
  $('#transcript').open = false;
  if (current === panel) renderPanel();
}
for (const button of document.querySelectorAll('[data-panel]')) {
  button.addEventListener('click', () => {
    state[current].draft = $('#prompt').value;
    current = button.dataset.panel;
    renderPanel();
  });
}
$('#new-chat').addEventListener('click', () => { if (!operation) resetPanel(current); });
function resizeComposer() {
  $('#prompt').style.height = 'auto';
  $('#prompt').style.height = `${Math.min($('#prompt').scrollHeight, 160)}px`;
}
$('#prompt').addEventListener('input', () => { state[current].draft = $('#prompt').value; resizeComposer(); });
$('#prompt').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!operation) $('#composer').requestSubmit();
  }
});

async function responseError(response) {
  try {
    const body = await response.json();
    return typeof body.detail === 'string' ? body.detail : `Request failed (HTTP ${response.status}). Check your input.`;
  } catch { return `Request failed (HTTP ${response.status}). Please try again.`; }
}
export async function streamSSE(url, init, onEvent, signal) {
  const response = await fetch(url, {method: 'POST', ...init, signal});
  if (!response.ok) throw new Error(await responseError(response));
  if (!response.body) throw new Error('Streaming is unavailable in this browser.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  function dispatch(block) {
    const data = block.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n');
    if (data) onEvent(JSON.parse(data));
  }
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, {stream: true});
      // The server uses LF; also accept CRLF event framing.
      buffer = buffer.replace(/\r\n/g, '\n');
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        dispatch(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
      }
      if (done) {
        if (buffer.trim()) dispatch(buffer);
        break;
      }
    }
  } finally {
    try { await reader.cancel(); } catch { /* Abort already closed the stream. */ }
    reader.releaseLock();
  }
}
async function health() {
  try {
    const response = await fetch('/api/health', {signal: AbortSignal.timeout(6000)});
    if (!response.ok) throw new Error('Health check failed');
    const status = await response.json();
    $('#model-name').textContent = status.model || 'Model not configured';
    $('#status-label').textContent = status.ok ? 'Server connected' : 'Server unavailable';
    $('#status-dot').className = `status-dot ${status.ok ? 'ok' : 'offline'}`;
    $('#offline-banner').hidden = status.ok;
    $('#status-label').title = status.detail || '';
  } catch {
    $('#status-label').textContent = 'Server unavailable';
    $('#status-dot').className = 'status-dot offline';
    $('#offline-banner').hidden = false;
  }
}
function startOperation(panel, kind) {
  operation = {panel, kind, controller: new AbortController()};
  updateControls();
  return operation.controller.signal;
}
function finishOperation(panel) {
  operation = null;
  updateControls();
  if (current === panel) {
    renderHistory();
    renderAttachments();
    if (!$('#prompt').disabled) $('#prompt').focus();
  }
}

$('#composer').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (operation) { operation.controller.abort(); return; }
  const prompt = $('#prompt').value.trim();
  if (!prompt || !prerequisites(current)) return;
  const panel = current;
  const panelState = state[panel];
  const signal = startOperation(panel, 'chat');
  panelState.draft = '';
  $('#prompt').value = '';
  resizeComposer();
  followBottom = true;
  const userView = appendView(panel, {kind: 'user', text: prompt});
  const replyView = appendView(panel, {kind: 'assistant', text: '', waiting: true});
  let answer = '';
  let usedTool = false;
  let terminal = false;
  let userAdded = false;
  let latestTool = null;
  try {
    let userMessage = {role: 'user', content: prompt};
    if (panel === 'image' && !panelState.messages.length) {
      const form = new FormData();
      form.append('file', panelState.extra.file);
      form.append('prompt', prompt);
      const response = await fetch('/api/image/prepare', {method: 'POST', body: form, signal});
      if (!response.ok) throw new Error(await responseError(response));
      userMessage = (await response.json()).message;
      userView.text = visibleText(userMessage);
    }
    panelState.messages.push(userMessage);
    userAdded = true;
    if (current === panel) renderAttachments();
    const messages = panel === 'audio' ? [panelState.extra.context, ...panelState.messages] : panelState.messages;
    await streamSSE(panel === 'agent' ? '/api/agent' : '/api/chat', {
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({messages}),
    }, (data) => {
      if (data.type === 'error') throw new Error(data.message);
      if (data.type === 'delta') { answer += data.text; updateAnswer(panel, replyView, answer); }
      if (data.type === 'done') terminal = true;
      if (data.type === 'tool_call') {
        usedTool = true;
        latestTool = {kind: 'tool', name: data.name, arguments: data.arguments, open: true};
        // Put tool activity before the answer, retaining it when switching panels.
        panelState.views.splice(panelState.views.indexOf(replyView), 0, latestTool);
        if (current === panel) renderHistory();
      }
      if (data.type === 'tool_result' && latestTool) {
        latestTool.result = data.content;
        if (current === panel) renderHistory();
      }
      if (data.type === 'final') {
        terminal = true;
        answer = data.content;
        if (!usedTool) replyView.note = '(no tool was used)';
        updateAnswer(panel, replyView, answer);
      }
    }, signal);
    if (!terminal) throw new Error('The connection ended before the response finished. Please try again.');
    if (!answer) updateAnswer(panel, replyView, 'The model returned an empty reply. Try asking again.');
  } catch (error) {
    if (error.name !== 'AbortError') {
      errorBubble(panel, error.message || 'Could not complete the request. Please try again.');
      void health();
    }
    if (!answer) panelState.views = panelState.views.filter((view) => view !== replyView);
    else replyView.note = error.name === 'AbortError' ? 'Stopped · partial answer' : 'Incomplete response';
    // A failed image preparation must not leave a phantom first image message.
    if (!userAdded) panelState.views = panelState.views.filter((view) => view !== userView);
  } finally {
    replyView.waiting = false;
    if (answer) panelState.messages.push({role: 'assistant', content: answer});
    finishOperation(panel);
  }
});

function attach(file) {
  if (!file || operation || !['image', 'audio'].includes(current)) return;
  const panel = current;
  const isImage = panel === 'image';
  const limit = (isImage ? 10 : 50) * 1024 * 1024;
  const valid = isImage ? ['image/jpeg', 'image/png', 'image/webp'].includes(file.type) : /\.(m4a|mp3|wav|ogg|aac)$/i.test(file.name);
  if (!valid) { errorBubble(panel, isImage ? 'Choose a JPG, PNG or WebP image.' : 'Choose an m4a, mp3, wav, ogg or aac recording.'); return; }
  if (file.size > limit || file.size === 0) { errorBubble(panel, `Choose a nonempty file up to ${limit / 1024 / 1024} MB.`); return; }
  resetPanel(panel);
  state[panel].extra.file = file;
  if (isImage) state[panel].extra.preview = URL.createObjectURL(file);
  renderPanel();
}
$('#choose-file').addEventListener('click', () => $('#file-input').click());
$('#file-input').addEventListener('change', (event) => attach(event.target.files[0]));
$('#remove-file').addEventListener('click', () => { if (!operation) resetPanel(current); });
const dropZone = $('#drop-zone');
for (const type of ['dragenter', 'dragover']) dropZone.addEventListener(type, (event) => {
  event.preventDefault();
  if (!operation) dropZone.classList.add('dragging');
});
for (const type of ['dragleave', 'drop']) dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); });
dropZone.addEventListener('drop', (event) => attach(event.dataTransfer.files[0]));
// Prevent accidental navigation when a file misses the upload card.
window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => event.preventDefault());

$('#transcribe').addEventListener('click', async () => {
  if (operation || !state.audio.extra.file) return;
  const panelState = state.audio;
  panelState.messages = [];
  panelState.views = [];
  panelState.extra.progress = [];
  delete panelState.extra.transcript;
  delete panelState.extra.context;
  const signal = startOperation('audio', 'transcribe');
  renderHistory();
  renderAttachments();
  const form = new FormData();
  form.append('file', panelState.extra.file);
  let completed = false;
  try {
    await streamSSE('/api/audio/transcribe', {body: form}, (data) => {
      if (data.type === 'error') throw new Error(data.message);
      if (data.type === 'progress') panelState.extra.progress.push(data);
      if (data.type === 'done') {
        if (!data.transcript) throw new Error('The model returned an empty transcript. Try another recording.');
        completed = true;
        panelState.extra.transcript = data.transcript;
        panelState.extra.context = data.context_message;
      }
      if (current === 'audio') renderAttachments();
    }, signal);
    if (!completed) throw new Error('Transcription ended before it finished. Please try again.');
  } catch (error) {
    if (error.name !== 'AbortError') { errorBubble('audio', error.message); void health(); }
  } finally { finishOperation('audio'); }
});

window.addEventListener('beforeunload', () => {
  operation?.controller.abort();
  for (const panelState of Object.values(state)) {
    if (panelState.extra.preview) URL.revokeObjectURL(panelState.extra.preview);
  }
});
renderPanel();
void health();
setInterval(health, 20000);
