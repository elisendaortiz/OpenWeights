import streamlit as st
from ui_pages.common import start_page, render_history, answer

start_page('chat', 'Chatbot', 'Ask a question, explore an idea, and keep the conversation going.', {'messages': []})
messages = st.session_state.chat_messages
render_history(messages)
if prompt := st.chat_input('What would you like to explore?'):
    messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    answer(messages)
