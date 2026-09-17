# AI-Powered Chat Service

Async FastAPI chat API (`GET /chat`) backed by a LangGraph agent. Clients send the latest `message` and an optional `session_id`; the server keeps the conversation transcript.

## Conversation history

History is stored **in process memory**: a Python `dict` keyed by `session_id`. Because the API is GET (one new `message` per request), the server holds state so the client does **not** resend the full transcript.

### How it works

1. Call `GET /chat?message=...` without `session_id`. The response includes a new UUID.
2. Pass that `session_id` on later `GET /chat` calls to continue the same conversation.
3. The graph **loads** prior messages before the agent node and **saves** the updated transcript after the agent replies.

Sessions are isolated: two `session_id` values never share messages.

### Limitations

This in-memory store is for **demo / prototype** use only. It is not a production session backend.

- **Lost on process restart.** Stopping uvicorn, crashing, or redeploying wipes every session. There is no disk, database, or Redis write.
- **Not shared across instances.** Each OS process has its own dict. Two uvicorn workers, two containers, or a load-balanced replica set will not see the same conversations.
- **Not a LangGraph checkpointer.** Restarting the app does not restore graph state.

### Production alternatives

Replace the in-memory dict with a shared store such as **Redis**, **Azure Cosmos DB**, or **PostgreSQL**, or persist graph state with a LangGraph **checkpointer** so history survives restarts and is visible to every instance.
