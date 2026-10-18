from contextlib import closing
import streamlit as st
from services.agent import run_agent_sync
from ui_pages.common import start_page, render_history, show_error

start_page('agent', 'Agent', 'Uses the MCP time server (uvx mcp-server-time). Try: What time is it in Tokyo?', {'messages': []})
messages = st.session_state.agent_messages
render_history(messages)
if prompt := st.chat_input('What time is it in Tokyo?'):
    messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    completed = False
    with st.chat_message('assistant'):
        used_tool = False
        panel = None
        try:
            with st.spinner('Working with the time agent...'), closing(run_agent_sync(list(messages))) as events:
                for event in events:
                    if event['type'] == 'tool_call':
                        used_tool = True
                        panel = st.expander(f"Tool call: {event['name']}", expanded=True)
                        panel.json(event['arguments'])
                    elif event['type'] == 'tool_result':
                        (panel or st).write(event['content'])
                    elif event['type'] == 'final':
                        if event['content']:
                            st.markdown(event['content'])
                            messages.append({'role': 'assistant', 'content': event['content']})
                            completed = True
                        else:
                            st.info('The model returned an empty reply. Try asking again.')
                        if not used_tool:
                            st.caption('(no tool was used)')
                    elif event['type'] == 'error':
                        st.error(event['message'])
        except Exception as exc:
            show_error(exc)
    if not completed:
        messages.pop()
