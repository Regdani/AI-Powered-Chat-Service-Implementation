# Deployment strategy

High-level CI/CD and cloud deployment notes for this prototype. The service is already containerized; this document describes how that image would move from laptop → CI → Azure.

## Goals

- Ship the same artifact locally and in the cloud (Docker image from the repo root `Dockerfile`).
- Keep secrets out of the image (no baked-in `.env`).
- Prefer a managed container host over VM babysitting.

## Local vs cloud

| | Local | Cloud (target) |
|---|--------|----------------|
| Run | `docker compose up --build` | Azure Container Apps (ACA) |
| Config | Optional `.env` via Compose `env_file` | ACA secrets / Key Vault references |
| Image | Built on the developer machine | Built in CI, stored in Azure Container Registry (ACR) |
| Port | `18080` | Ingress → container port `18080` |
| Health | `GET /health` (Dockerfile `HEALTHCHECK`) | ACA HTTP probe on `/health` |

Without `.env`, the app defaults to `LLM_PROVIDER=mock`. That is useful for smoke runs; production would set a real provider and keys via the platform secret store.

## Target architecture (Azure)

```text
GitHub (source)
    │
    ▼
GitHub Actions CI
    │  pytest + docker build
    │  (future CD: push image)
    ▼
Azure Container Registry
    │
    ▼
Azure Container Apps  ── secrets ──► Key Vault (API keys)
    │
    └── ingress :443 → container :18080
```

**Suggested Azure pieces**

| Resource | Role |
|----------|------|
| Resource group | Scope for all env resources |
| ACR | Store `ai-chat-service:<git-sha>` images |
| Log Analytics + Container Apps Environment | Runtime + logs |
| Container App | 1 revision, port **18080**, min replicas **1** |
| Key Vault (or ACA secret store) | `GOOGLE_API_KEY`, Azure OpenAI keys, etc. |

**Container App settings (concept)**

- Image: `acr.azurecr.io/ai-chat-service:<sha>`
- Ingress: external HTTPS → target port `18080`
- Env (non-secret): `APP_HOST=0.0.0.0`, `APP_PORT=18080`, `LLM_PROVIDER=gemini|azure`, `GEMINI_MODEL`, MCP URL/transport
- Secrets: provider API keys only
- Health probe: HTTP `GET /health`

App Service is a valid alternative (Linux container), but Container Apps fits a single long-running HTTP API with less App Service plan overhead for a prototype.

## CI today

Workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

On every `push` / `pull_request` to `master`:

1. **test** — Python 3.11, `pip install -r requirements.txt`, `pytest`
2. **docker** — `docker build -t ai-chat-service:$GITHUB_SHA .` (no registry push)

This proves the image builds and the unit/integration suite (mock LLM) passes without Azure credentials.

## CD (not automated in this repo)

A production-shaped pipeline would extend CI after green builds:

1. Authenticate to ACR (`azure/login` + `docker login`).
2. Tag and push `ai-chat-service:$GITHUB_SHA` (and optionally `:latest` on `master`).
3. Update the Container App to the new image (CLI, Bicep/Terraform, or `Azure/container-apps-deploy-action`).
4. Gate deploy on `master` or version tags (`v*`); keep PRs at build+test only.

Secrets for CD (ACR, Azure subscription) live in GitHub Actions secrets / OIDC federated identity — never in the Dockerfile or git.

Azure DevOps Pipelines can mirror the same stages (`pytest` → `docker build` → `ACR push` → `ACA deploy`) if the org standard is ADO instead of GitHub Actions.

## Secrets and configuration

Aligns with the current app:

- Image does **not** copy `.env` (see `.dockerignore`).
- Runtime reads env vars (`LLM_PROVIDER`, Azure/Gemini keys, MCP settings).
- Fail-fast: selecting `gemini` / `azure` without required keys aborts startup (no silent mock fallback).

| Kind | Examples | Where in Azure |
|------|----------|----------------|
| Non-secret | `LLM_PROVIDER`, `APP_PORT`, `GEMINI_MODEL`, `MCP_WEATHER_URL` | Container App env |
| Secret | `GOOGLE_API_KEY`, `AZURE_OPENAI_API_KEY`, … | ACA secrets or Key Vault references |

## Session memory and scale

History is **in-process** (`InMemorySessionStore`). Consequences for deploy:

- Prefer **one replica** (or sticky sessions) so `session_id` stays on the same instance.
- Rolling restart or a second replica **drops** or splits conversation state.
- A later production step would externalize history (e.g. Redis / Cosmos) and enable horizontal scale.

Document this limit explicitly when demoing multi-instance ideas.

## IaC ideas (Terraform)

Illustrative resource set (not applied in this prototype):

```hcl
# Sketch only — names and wiring for discussion
# azurerm_resource_group
# azurerm_container_registry
# azurerm_log_analytics_workspace
# azurerm_container_app_environment
# azurerm_container_app
#   - template.container.image = "${acr_login_server}/ai-chat-service:${var.image_tag}"
#   - ingress.target_port      = 18080
#   - secret / Key Vault refs for LLM keys
# azurerm_key_vault (+ access policy / RBAC for the Container App identity)
```

Terraform (or Bicep) would own resource identity and wiring; the pipeline would only pass `image_tag` (= git SHA). Applying that stack is out of scope for the assignment submission unless an Azure subscription is available.

## Quick reference

```bash
# Local (parity with cloud artifact)
docker compose up --build
# → http://localhost:18080/health
# → http://localhost:18080/docs

# CI-equivalent checks
pytest
docker build -t ai-chat-service:local .
```

See also: root [README](../README.md) (API, providers, Docker quick start).
