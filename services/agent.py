"""Time agent: a chat where the model answers time questions by calling MCP tools.

Run from the repo root:
  uv run python agent_timezone_service.py

Ask things like "What time is it in Tokyo?", "convert 3pm Singapore to London
time" or "what time is it in the capital of Peru?", then follow up ("and in
New York?", "what's the difference between them?").

How it works:
  1. The script starts the MCP time server (`uvx mcp-server-time`) as a
     subprocess and talks to it over stdin/stdout with the MCP client. It lists
     the server's tools (get_current_time, convert_time) and prints their names.
  2. Each MCP tool is converted to the OpenAI `tools` format (name, description,
     JSON schema) and attached to every request, together with a system prompt
     that tells the model to use the tools and IANA timezone names.
  3. If the reply contains `tool_calls`, the script runs each call through MCP
     (`session.call_tool`), appends the assistant message and a `role: "tool"`
     message with the result (matching `tool_call_id`), and calls the model
     again. This repeats until the model replies without a tool call, up to
     MAX_ITERATIONS rounds per question, so a confused model cannot loop forever.
     That loop is what makes it an agent.
  4. If a tool call fails (unknown tool, invalid JSON arguments, an error from
     the server), the error text is sent back to the model as the tool result,
     so it can retry with corrected arguments.
  5. Every tool call and its result is printed, e.g.
       → get_current_time(timezone="Asia/Tokyo")
         ← {"timezone": "Asia/Tokyo", ...}
     so you can see when a small model picks the wrong tool, mangles a timezone
     name or ignores the tools. If the model answers without calling any tool,
     the script prints "(no tool used)". That is normal for follow-ups that need
     no tool, but a problem for a plain time question.
  6. The full `messages` history is kept, so follow-up questions have context.

More MCP servers can be added as extra entries in SERVERS. Their tools are
merged into one list and each call is routed to the server that owns the tool.

Requires uvx (it comes with uv) to download and run the MCP server on demand,
and a model with tool-calling support.

Settings come from a `.env` file (see `.env.example`):
  BASE_URL  OpenAI-compatible endpoint
  MODEL     model name served by that endpoint (must support tools)
  API_KEY   API key (Ollama ignores it, but the client library requires one)
Nothing is hardcoded, so the same code can point at another server just by
editing `.env`. Real environment variables take priority over the file.

Quit with: 'exit', 'quit' or Ctrl+C.
"""

import asyncio
from collections.abc import Iterator
import json
import shutil
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, stdio_client
from openai import APIError, AsyncOpenAI

# MCP servers to start. To add another agent (e.g. stock market data), add one more entry.
SERVERS = {
    "time": StdioServerParameters(command="uvx", args=["mcp-server-time"]),
}
MAX_ITERATIONS = 5  # max model calls per user question
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to time tools. "
    "For any question about the current time or about converting a time between "
    "timezones, call the tools instead of answering from memory. "
    "Always use IANA timezone names such as Asia/Tokyo, America/New_York, "
    "Europe/London or America/Lima: translate cities and countries into the "
    "matching IANA name yourself. "
    "Then answer briefly in plain language using the tool results."
)

from services.config import require_env, SAMPLING_PARAMETERS


async def connect_servers(stack: AsyncExitStack, quiet=False) -> tuple[list[dict], dict[str, ClientSession]]:
    """Start every MCP server; return the tools in OpenAI format and a tool name -> session map."""
    for command in {params.command for params in SERVERS.values()}:
        if shutil.which(command) is None:
            raise SystemExit(f"{command} not found. It comes with uv: https://docs.astral.sh/uv/")

    tools: list[dict] = []
    routes: dict[str, ClientSession] = {}

    for server_name, params in SERVERS.items():
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listing = await session.list_tools()
        except Exception as err:
            raise SystemExit(
                f"Could not start MCP server '{server_name}' "
                f"({params.command} {' '.join(params.args)}): {err}"
            )

        if not quiet:
            print(f"[{server_name}] tools: {', '.join(tool.name for tool in listing.tools)}")
        for tool in listing.tools:
            if tool.name in routes:
                raise SystemExit(f"Tool name '{tool.name}' is provided by more than one MCP server.")
            routes[tool.name] = session
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
            )

    return tools, routes


async def execute_tool(routes: dict[str, ClientSession], name: str, raw_arguments: str) -> str:
    """Run one tool call through MCP. Errors are returned as text so the model can retry."""
    try:
        arguments = json.loads(raw_arguments) if raw_arguments else {}
    except json.JSONDecodeError:
        print(f"→ {name}({raw_arguments})  ✗ invalid JSON arguments")
        return f"Error: the arguments were not valid JSON: {raw_arguments}"
    if not isinstance(arguments, dict):
        print(f"→ {name}({raw_arguments})  ✗ arguments are not a JSON object")
        return "Error: the arguments must be a JSON object."

    shown = ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in arguments.items())
    print(f"→ {name}({shown})")

    session = routes.get(name)
    if session is None:
        print(f"  ✗ unknown tool '{name}'")
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(routes)}."

    try:
        result = await session.call_tool(name, arguments)
    except Exception as err:
        print(f"  ✗ {err}")
        return f"Error: calling the tool failed: {err}"

    blocks = getattr(result, "content", None) or []
    text = "\n".join(block.text for block in blocks if getattr(block, "type", None) == "text")
    if not text and getattr(result, "structured_content", None) is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)
    if getattr(result, "is_error", False):
        print(f"  ✗ {text}")
        return f"Error: {text}"
    print(f"  ← {text}")
    return text


async def run_turn(
    client: AsyncOpenAI,
    model: str,
    tools: list[dict],
    routes: dict[str, ClientSession],
    messages: list,
) -> None:
    """Answer the last user message, running tool calls until the model replies in plain text."""
    for iteration in range(MAX_ITERATIONS):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            if iteration == 0:
                print("(no tool used)")
            messages.append({"role": "assistant", "content": message.content or ""})
            print(f"Bot: {message.content}\n")
            return

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            result = await execute_tool(routes, call.function.name, call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    print(f"Stopped after {MAX_ITERATIONS} rounds of tool calls without a final answer.\n")


async def main() -> None:
    base_url = require_env("BASE_URL")
    model = require_env("MODEL")
    api_key = require_env("API_KEY")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async with AsyncExitStack() as stack:
        tools, routes = await connect_servers(stack)

        messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]
        print(f"Model: {model} at {base_url}")
        print("Ask about the time. Type 'exit' or press Ctrl+C to quit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            history_length = len(messages)
            messages.append({"role": "user", "content": user_input})
            try:
                await run_turn(client, model, tools, routes, messages)
            except APIError as err:
                del messages[history_length:]  # drop this failed turn from the history
                print(
                    f"The model server returned an error: {err}\n"
                    "Check that MODEL supports tool calling (ollama show <model>).\n"
                )



MCP_SERVER = SERVERS['time']


def mcp_tools_to_openai(tools) -> list[dict]:
    return [{"type": "function", "function": {
        "name": tool.name, "description": tool.description or "",
        "parameters": tool.input_schema,
    }} for tool in tools]


async def _run_agent(messages, max_iterations, emit):
    """Keep MCP contexts, model requests and teardown in the same async task."""
    if shutil.which('uvx') is None:
        raise RuntimeError('uvx not found. Install uv to use the time agent.')
    history = [dict(message) for message in messages]
    if not any(message.get('role') == 'system' for message in history):
        history.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    async with AsyncOpenAI(base_url=require_env('BASE_URL'), api_key=require_env('API_KEY')) as client:
        async with stdio_client(MCP_SERVER) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = mcp_tools_to_openai((await session.list_tools()).tools)
                names = {tool['function']['name'] for tool in tools}
                for _ in range(max_iterations):
                    response = await client.chat.completions.create(
                        model=require_env('MODEL'), messages=history, tools=tools, **SAMPLING_PARAMETERS)
                    message = response.choices[0].message
                    if not message.tool_calls:
                        emit({'type': 'final', 'content': message.content or ''})
                        return
                    history.append(message.model_dump(exclude_none=True))
                    for call in message.tool_calls:
                        name = call.function.name
                        try:
                            arguments = json.loads(call.function.arguments or '{}')
                            if not isinstance(arguments, dict):
                                raise ValueError('The arguments must be a JSON object.')
                        except (ValueError, TypeError) as exc:
                            arguments = {}
                            result = f'Error: invalid JSON arguments: {exc}'
                            emit({'type': 'tool_call', 'name': name, 'arguments': arguments})
                        else:
                            emit({'type': 'tool_call', 'name': name, 'arguments': arguments})
                            try:
                                if name not in names:
                                    raise ValueError(f'Unknown tool: {name}')
                                reply = await session.call_tool(name, arguments)
                                result = '\n'.join(block.text for block in reply.content if getattr(block, 'type', None) == 'text')
                                if not result and reply.structured_content is not None:
                                    result = json.dumps(reply.structured_content, ensure_ascii=False)
                                if reply.is_error:
                                    result = f'Error: {result}'
                            except Exception as exc:
                                result = f'Error: calling the tool failed: {exc}'
                        emit({'type': 'tool_result', 'name': name, 'content': result})
                        history.append({'role': 'tool', 'tool_call_id': call.id, 'content': result})
                emit({'type': 'error', 'message': f'Stopped after {max_iterations} rounds of tool calls without a final answer.'})


def run_agent_sync(messages: list[dict], max_iterations: int = MAX_ITERATIONS) -> Iterator[dict]:
    from queue import Queue
    from threading import Event, Thread

    events = Queue()
    finished = object()
    stop = Event()

    async def runner():
        # A watcher cancels the single task owning all MCP contexts, including
        # while it awaits a model response. It never closes those contexts itself.
        owner = asyncio.current_task()
        async def watch():
            while not stop.is_set():
                await asyncio.sleep(0.05)
            owner.cancel()
        watcher = asyncio.create_task(watch())
        try:
            if not stop.is_set():
                await _run_agent(messages, max_iterations, events.put)
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    def worker():
        try:
            asyncio.run(runner())
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            if isinstance(exc, APIError):
                reason = f'Model request failed ({type(exc).__name__}). Check BASE_URL and MODEL tool support.'
            else:
                reason = f'Agent failed ({type(exc).__name__}). Check uvx and the MCP time server.'
            events.put({'type': 'error', 'message': reason})
        finally:
            events.put(finished)

    thread = Thread(target=worker, daemon=True, name='openweights-agent')
    thread.start()
    try:
        while True:
            event = events.get()
            if event is finished:
                break
            yield event
    finally:
        stop.set()
        thread.join(timeout=3)
