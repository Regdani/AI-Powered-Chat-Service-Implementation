# AI-Powered Chat Service

Async FastAPI chat API backed by a LangGraph agent. Clients send the latest `message` and an optional `session_id`; the server keeps the conversation transcript in memory.

Interactive docs: [Swagger UI](http://localhost:18080/docs) · [ReDoc](http://localhost:18080/redoc)

Default bind: `http://localhost:18080` (`APP_HOST` / `APP_PORT` in `.env`).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # edit as needed
python -m app
```

This starts uvicorn on `APP_HOST` / `APP_PORT` from `.env` (defaults: `0.0.0.0:18080`). No `--port` flag needed.

Equivalent explicit form (if you prefer the CLI):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 18080
```

With no real provider configured, `LLM_PROVIDER=mock` is the default (deterministic echo + fixed joke; no tool calls).

## Docker

Build and run the API in a container (port **18080**):

```bash
docker compose up --build
```

- Loads `.env` automatically when present; if missing, the app uses built-in defaults (`LLM_PROVIDER=mock`, `APP_HOST=0.0.0.0`, `APP_PORT=18080`).
- The image does **not** bake in `.env` — secrets stay on the host via Compose `env_file`.
- Health: `GET /health` (also used by the image `HEALTHCHECK`).
- Docs: http://localhost:18080/docs

Equivalent without Compose:

```bash
docker build -t ai-chat-service .
docker run --rm -p 18080:18080 --env-file .env ai-chat-service   # with .env
docker run --rm -p 18080:18080 ai-chat-service                   # defaults / mock
```

## Deployment

High-level CI/CD and Azure Container Apps strategy: [docs/deployment.md](docs/deployment.md).

CI (GitHub Actions on `master`): `pytest` + `docker build` — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## HTTP APIs

### `GET /health`

Liveness probe. Returns `200` when the API process is up. Independent of the LLM provider and of the remote weather MCP.

**Response**

```json
{ "status": "ok" }
```

### `GET /chat`

Runs one turn of the LangGraph agent:

`load_history` → `agent` ⇄ `tools` → `translate` → `save_history`

The agent echoes the user text and adds a short joke. Tools (Megafon, weather) run only when the LLM emits tool calls. `language` is applied by a separate translate node (not by the agent).

#### Query parameters

| Parameter     | Required | Default | Description |
|---------------|----------|---------|-------------|
| `message`     | yes      | —       | User text sent to the agent (non-blank). |
| `language`    | no       | `en`    | Desired reply language code (e.g. `en`, `hu`). Applied inside the graph; not returned in the body. English skips the extra LLM translate call. |
| `session_id`  | no       | new UUID| Continue an existing conversation. Omit (or blank) to start a new session. |

#### Response (`200`)

```json
{
  "session_id": "3f1c8a6e-9b2d-4e71-a4c8-1d2e3f4a5b6c",
  "answer": "You asked: \"Hello\". … joke …"
}
```

- `session_id` — value to reuse on later turns (client-supplied or newly generated).
- `answer` — agent reply for this turn (echo + joke, plus any tool output; translated when `language` ≠ `en`).

#### Errors

| Status | When |
|--------|------|
| `422`  | Missing or invalid query parameters. |
| `502`  | LLM / graph failure, or empty agent reply. |
| `503`  | Chat graph not available. |

#### Examples

```bash
# New session (English)
curl "http://localhost:18080/chat?message=Hello"

# Hungarian reply
curl "http://localhost:18080/chat?message=Szia&language=hu"

# Continue a session
curl "http://localhost:18080/chat?message=What%20did%20I%20just%20say%3F&session_id=YOUR_SESSION_ID"
```

## Configure Gemini

1. Copy `.env.example` to `.env` (if you have not already).
2. Set:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-google-api-key
GEMINI_MODEL=gemini-3.1-flash-lite
```

3. Restart uvicorn. Startup **fails fast** if `LLM_PROVIDER=gemini` and `GOOGLE_API_KEY` is missing — there is no silent fallback to mock.

`GEMINI_MODEL` defaults to `gemini-3.1-flash-lite` when unset. Other supported providers: `mock` (default) and `azure` — see `.env.example` for Azure variables.

## Configure Azure OpenAI

```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

Startup fails fast if any of the required Azure variables are missing.

With Gemini or Azure, the model can call tools (`megafon`, `get_weather`) according to the system prompt. Joke wording varies by model; small flash models often reuse common dad jokes.

## Announce (Megafon tool)

To shout text, include the `Announce:` prefix in the message. The **megafon** tool uppercases the text after that marker (case-insensitive).

```bash
curl "http://localhost:18080/chat?message=Announce:%20hello%20world&session_id=megafon-demo"
```

Expected behaviour with a **real** LLM (`gemini` or `azure`):

1. The agent calls `megafon` with the announcement text.
2. The tool returns e.g. `HELLO WORLD`.
3. The reply includes the shouted result plus a short joke.

Example log:

```
INFO [app.tools.megafon] Megafon announce input='Announce: hello world' output='HELLO WORLD'
INFO [app.graph.nodes] Agent tool calls session_id=megafon-demo names=['megafon']
INFO [app.graph.nodes] Agent reply session_id=megafon-demo answer=You said: HELLO WORLD. …
```

**Mock mode:** the default mock LLM does **not** detect `Announce:` and does **not** emit tool calls. `Announce: hello` returns plain echo + fixed joke. That is expected. The tool loop is covered by tests with a scripted LLM and by a live `LLM_PROVIDER`.

## Weather MCP

The agent looks up weather through a **local wrapper tool** (`get_weather`) that proxies to the public Open-Meteo MCP at `https://weather.chukai.io/mcp` (no API key).

### Configuration

```env
MCP_WEATHER_URL=https://weather.chukai.io/mcp
MCP_WEATHER_TRANSPORT=streamable_http
```

Startup only **configures** the MCP client (URL + transport). It does **not** open a remote session and does **not** call `get_tools()` on the network.

### Runtime

1. User asks about **current** weather (e.g. “What’s the weather in Budapest?”).
2. With a real LLM, the agent calls `get_weather` with the location.
3. On the **first** call, the client connects lazily (handshake → remote geocode → forecast), then returns a short summary to the agent.
4. If the remote is down, the tool returns an error string to the agent; the API process, `GET /health`, echo+joke, Megafon, and history keep working.

```bash
curl "http://localhost:18080/chat?message=What%27s%20the%20weather%20like%20in%20Budapest%3F"
```

**Mock mode:** weather questions do not trigger tool calls (echo + joke only). Covered by scripted tests and a live provider demo.

**History rule:** when the user asks about a *previous* weather answer, the agent must use only the current session transcript — it should not call MCP to reconstruct prior replies. If that turn never happened in this session, it should say it does not have that information.

## Conversation history

History is stored **in process memory**: a Python `dict` keyed by `session_id`. Because the API is GET (one new `message` per request), the server holds state so the client does **not** resend the full transcript.

### How it works

1. Call `GET /chat?message=...` without `session_id`. The response includes a new UUID.
2. Pass that `session_id` on later `GET /chat` calls to continue the same conversation.
3. The graph **loads** prior messages before the agent node and **saves** the updated transcript after the agent replies (and after translation).

Sessions are isolated: two `session_id` values never share messages. Asking “according to your previous answer?” in a **new** session correctly has no prior answer — even if another session already discussed weather.

### Limitations

This in-memory store is for **demo / prototype** use only. It is not a production session backend.

- **Lost on process restart.** Stopping uvicorn, crashing, or redeploying wipes every session. There is no disk, database, or Redis write.
- **Not shared across instances.** Each OS process has its own dict. Two uvicorn workers, two containers, or a load-balanced replica set will not see the same conversations.
- **Not a LangGraph checkpointer.** Restarting the app does not restore graph state.

### Production alternatives

Replace the in-memory dict with a shared store such as **Redis**, **Azure Cosmos DB**, or **PostgreSQL** so history survives restarts and is visible to every instance.
