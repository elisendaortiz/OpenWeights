import streamlit as st
from services.image import prepare_image, build_image_message
from ui_pages.common import start_page, render_history, answer, show_error

start_page('image', 'Images', 'Upload a photo or screenshot, then ask questions about it.',
           {'messages': [], 'file_id': None, 'data_url': None})
uploaded = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png', 'webp'],
                            key=f'image_upload_{st.session_state.image_upload_generation}')
file_id = uploaded.file_id if uploaded else None
if file_id != st.session_state.image_file_id:
    st.session_state.image_messages = []
    st.session_state.image_data_url = None
    st.session_state.image_file_id = file_id
    if uploaded:
        try:
            st.session_state.image_data_url = prepare_image(uploaded.getvalue())
        except ValueError as exc:
            show_error(exc)
if uploaded and st.session_state.image_data_url:
    # Display the normalized, bounded-size image, respecting EXIF orientation.
    st.image(st.session_state.image_data_url)
elif uploaded:
    st.error('Not a valid image file. Upload a JPG, PNG or WebP image.')
messages = st.session_state.image_messages
render_history(messages)
if prompt := st.chat_input('Describe this image.', disabled=not st.session_state.image_data_url):
    messages.append(build_image_message(st.session_state.image_data_url, prompt) if not messages
                    else {'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    answer(messages)
