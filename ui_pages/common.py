"""Small shared UI helpers; all conversation state belongs to a page."""
from contextlib import closing
import streamlit as st
from openai import APIConnectionError, APIStatusError, APITimeoutError
from services.config import check_server
from services.chat import message_text, stream_chat


@st.cache_data(ttl=10)
def server_status():
    return check_server()


def start_page(prefix, title, caption, defaults):
    for key, value in defaults.items():
        st.session_state.setdefault(f'{prefix}_{key}', value)
    st.session_state.setdefault(f'{prefix}_upload_generation', 0)
    st.title(title)
    st.caption(caption)
    if st.button('New conversation', key=f'{prefix}_new'):
        generation = st.session_state[f'{prefix}_upload_generation'] + 1
        for key in list(st.session_state):
            if key.startswith(prefix + '_') and key != f'{prefix}_new':
                del st.session_state[key]
        st.session_state[f'{prefix}_upload_generation'] = generation
        st.rerun()
    ok, reason = server_status()
    if not ok:
        st.error(f'{reason} Is Ollama running?')
        st.stop()


def render_history(messages):
    for message in messages:
        with st.chat_message(message['role']):
            st.markdown(message_text(message))


def show_error(exc):
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        server_status.clear()
        st.error('Could not reach the model server. Is Ollama running? Check BASE_URL and try again.')
    elif isinstance(exc, APIStatusError):
        st.error(f'The model server rejected the request (HTTP {exc.status_code}). Check MODEL and whether the server supports this input type.')
    elif isinstance(exc, (ValueError, RuntimeError)):
        st.error(str(exc))
    else:
        st.error(f'The request could not be completed ({type(exc).__name__}). Check your input and try again.')


def answer(messages, context=None):
    """Commit an assistant turn only after successful streaming; allow retry."""
    with st.chat_message('assistant'):
        try:
            with st.spinner('Thinking...'), closing(stream_chat([*context, *messages] if context else messages)) as tokens:
                reply = st.write_stream(tokens)
            if reply:
                messages.append({'role': 'assistant', 'content': reply})
            else:
                messages.pop()
                st.info('The model returned an empty reply. Try asking again.')
        except Exception as exc:
            messages.pop()
            show_error(exc)
