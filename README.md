# OpenWeights
Library of examples of how to serve with Open Weight models 

Each use case below is a small, self-contained example. More will be added over time.

## Use case 1: Chatbots

A terminal chatbot that talks to an open-weights model through an OpenAI-compatible API. It runs locally with [Ollama](https://ollama.com) and works with other servers, such as vLLM in the cloud, by changing only the settings.

| Script | What it does |
| --- | --- |
| `chat_service.py` | Chat loop that keeps the conversation history. The reply appears all at once. |
| `chat_streaming_service.py` | Same chat loop, but tokens are printed as they arrive. |

### Setup

```bash
uv sync                      # install dependencies
cp .env.example .env         # then edit .env if needed
```

`.env` holds `BASE_URL`, `MODEL` and `API_KEY`. The defaults point at a local Ollama server. Make sure Ollama is running and the model is pulled (for example `ollama pull qwen2.5:1.5b`).

### Run

```bash
uv run chat_service.py
uv run chat_streaming_service.py
```

Type `exit` or press Ctrl+C to quit.
