# server2 — connection parameters

| Field | Value |
|-------|--------|
| **Role** | Second Phase 3 **mediator / GPU host** (CUDA execution for guest vGPU path). |
| **Address** | `10.25.33.20` |
| **SSH user** | `root` |
| **Authentication** | Password-based SSH (same class of access as Phase 3 `connect_host.py` expects). |

## Environment variables (`phase3/`)

Run from **`server2/phase3/`** (or set `PYTHONPATH`). `connect_host.py` reads `vm_config.py`:

- `MEDIATOR_HOST` — default in **`vm_config.py`** is still **`10.25.33.10`** (first host). For server2 use **`export MEDIATOR_HOST=10.25.33.20`** or replace `vm_config.py` with **`../vm_config_server2.py`** (copy/rename).
- `MEDIATOR_USER` — default: `root`
- `MEDIATOR_PASSWORD` — **must be set** in your shell for non-interactive use (do **not** commit passwords to git).

Example:

```bash
export MEDIATOR_HOST=10.25.33.20
export MEDIATOR_USER=root
export MEDIATOR_PASSWORD='your-secret'
cd server2/phase3 && python3 connect_host.py 'grep module-load /tmp/mediator.log | tail -20'
```

## Host layout (expected)

Align with your on-host Phase 3 tree (e.g. `/root/phase3` or your chosen `REMOTE_PHASE3`). `deploy_cuda_executor_to_host.py` uses `REMOTE_PHASE3` from the environment (default `/root/phase3`).

**Credential note:** Store the root password in a password manager or `~/.netrc`-style workflow you control; avoid checking secrets into the repository.
