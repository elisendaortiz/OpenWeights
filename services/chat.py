"""Shared text generation and terminal chat helpers."""
from collections.abc import Iterator
from services.config import BASE_URL, MODEL, SAMPLING_PARAMETERS, get_client


def stream_chat(messages: list[dict]) -> Iterator[str]:
    with get_client().chat.completions.create(
        model=MODEL, messages=messages, stream=True, **SAMPLING_PARAMETERS
    ) as stream:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def message_text(message: dict) -> str:
    content = message.get('content')
    if isinstance(content, str):
        return content
    return '\n'.join(part.get('text', '') for part in content or [] if part.get('type') == 'text')


def stream_reply(client, model, messages: list) -> str:
    print('Bot: ', end='', flush=True)
    tokens = []
    for token in stream_chat(messages):
        print(token, end='', flush=True)
        tokens.append(token)
    print('\n')
    return ''.join(tokens)


def main(streaming=False):
    client = get_client()
    messages = []
    print(f"Chatting with {MODEL} at {BASE_URL}. Type 'exit' or press Ctrl+C to quit.\n")
    while True:
        try:
            user_input = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {'exit', 'quit'}:
            break
        messages.append({'role': 'user', 'content': user_input})
        if streaming:
            reply = stream_reply(client, MODEL, messages)
        else:
            response = client.chat.completions.create(model=MODEL, messages=messages, **SAMPLING_PARAMETERS)
            reply = response.choices[0].message.content
            print(f'Bot: {reply}\n')
        messages.append({'role': 'assistant', 'content': reply})
