from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import streamlit as st
from services.audio import iter_transcribe, build_transcript_context, require_ffmpeg
from ui_pages.common import start_page, render_history, answer, show_error

start_page('audio', 'Audio', 'Transcribe a recording, then explore what was said.',
           {'messages': [], 'transcript': '', 'file_id': None})
st.caption('Answers come from the transcript only; tone and non-speech sounds are not heard.')
uploaded = st.file_uploader('Upload audio', type=['m4a', 'mp3', 'wav', 'ogg', 'aac'],
                            key=f'audio_upload_{st.session_state.audio_upload_generation}')
file_id = uploaded.file_id if uploaded else None
if file_id != st.session_state.audio_file_id:
    st.session_state.audio_file_id = file_id
    st.session_state.audio_messages = []
    st.session_state.audio_transcript = ''
if st.button('Transcribe', disabled=not uploaded):
    st.session_state.audio_messages = []
    st.session_state.audio_transcript = ''
    with st.status('Transcribing...', expanded=True) as status:
        try:
            require_ffmpeg()
            suffix = Path(uploaded.name).suffix.lower()
            if suffix not in {'.m4a', '.mp3', '.wav', '.ogg', '.aac'}:
                raise ValueError('Unsupported audio file type.')
            with TemporaryDirectory(prefix='openweights-upload-') as directory:
                source = Path(directory) / ('input' + suffix)
                source.write_bytes(uploaded.getvalue())
                with closing(iter_transcribe(source)) as events:
                    for event in events:
                        if event['type'] == 'progress':
                            st.write(f"Chunk {event['index']}/{event['total']}")
                            st.write(event['text'])
                        elif event['type'] == 'done':
                            st.session_state.audio_transcript = event['transcript']
            if not st.session_state.audio_transcript:
                raise RuntimeError('The model returned an empty transcript. Try another recording.')
            status.update(label='Transcription complete', state='complete', expanded=False)
        except Exception as exc:
            status.update(label='Transcription failed', state='error')
            show_error(exc)
transcript = st.session_state.audio_transcript
if transcript:
    with st.expander('Transcript'):
        st.write(transcript)
messages = st.session_state.audio_messages
render_history(messages)
if prompt := st.chat_input('Summarize this audio.', disabled=not transcript):
    messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    answer(messages, context=[build_transcript_context(transcript)])
