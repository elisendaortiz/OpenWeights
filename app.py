"""Local Streamlit entrypoint: uv run streamlit run app.py."""
import streamlit as st
from services.config import MODEL
from ui_pages.common import server_status

st.set_page_config(page_title="OpenWeights Playground", page_icon="🧪", layout="centered")
nav = st.navigation([
    st.Page('ui_pages/chat_page.py', title='Chatbot', icon='💬', default=True),
    st.Page('ui_pages/image_page.py', title='Images', icon='🖼️'),
    st.Page('ui_pages/audio_page.py', title='Audio', icon='🎧'),
    st.Page('ui_pages/agent_page.py', title='Agent', icon='🕒'),
])
with st.sidebar:
    st.title('🧪 OpenWeights')
    st.caption('Your local model playground')
    st.markdown('**Model**')
    st.code(MODEL or 'Not configured', language=None)
    ok, reason = server_status()
    if ok:
        st.success('✓ Server connected')
    else:
        st.error(reason)
        st.caption('Is Ollama running?')
    st.caption('One model · Four ways to explore')
nav.run()
