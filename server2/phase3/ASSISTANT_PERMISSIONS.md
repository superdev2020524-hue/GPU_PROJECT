# Assistant permissions (Phase 3)

*Clarified: Mar 18, 2026 - after host/VM modifications for VGPU display name (lspci) and rebuild.*

---

## Server 2 boundary

For the current Server 2 mission, follow
**`SERVER2_ISOLATION_AND_MISSION_RULES.md`** and treat:

- **`server2/phase3/`** as the editable Server 2 registry
- root **`phase3/`** as protected Server 1 work

These permissions do **not** authorize edits to root `phase3/` for Server 2.

## Host

- **Allowed:** Check host logs (e.g. `/tmp/mediator.log`, daemon.log for stub) and **read file contents** for investigation and verification.
- **Allowed (when you explicitly grant it):** Use read-only access to **stage artifacts from dom0 toward the VM** (e.g. you copy dom0 → VM, or authorize scripts that pull from paths you expose). **Copying onto dom0** remains **yours** unless you say otherwise.
- **Not allowed:** No **editing** of host files without your explicit go-ahead. No building / `make` / restart mediator on the host unless you ask for a documented exception.

Host-side fixes are documented; you apply them on the host when edits/builds are required.

---

## VM (Server 2 target VM)

- **Full authority:** Run commands, configure, deploy guest artifacts, edit VM files, read VM logs, rebuild and install software on the VM (e.g. ollama.bin, guest shims), restart services.

---

## Summary

| Scope   | Logs / read files | Edit / deploy / build / restart |
|---------|-------------------|----------------------------------|
| **Host** | Yes               | No                               |
| **VM**   | Yes               | Yes (full)                       |

---

## Role and anti-coupling

- **ASSISTANT_ROLE_AND_ANTICOUPLING.md** — On error: first search PHASE3 for past resolutions; verify no negative impact on previously working behavior (e.g. GPU mode, runner env); for VM build always check `/usr/local/go/bin/go version` before concluding the VM cannot build.
- **SERVER2_ISOLATION_AND_MISSION_RULES.md** — mandatory Server 2 mission registration, path boundary, and non-mixing rule.
- **SYSTEMATIC_ERROR_TRACKING_PLAN.md** — **Mandatory** operational procedure: checkpoints **A–D**, error registry **E1/E2/…**, gates before long runs, and required fields in assistant status updates (see **ASSISTANT_ROLE_AND_ANTICOUPLING.md §5**).

---

## Assistant operational obligations (Phase 3 / Phase 1)

When triaging errors or advancing the Phase 1 milestone, the assistant **must**:

1. Follow **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`** and **`ASSISTANT_ROLE_AND_ANTICOUPLING.md` §5** (not ad-hoc “run and wait”).
2. Use **`connect_vm.py`** and **`connect_host.py`** (read-only on host) to capture **Checkpoint A–C** evidence before recommending long generates.
3. Record material changes in **`ERROR_TRACKING_STATUS.md`** (short dated note + registry state).

This does **not** grant host edit/build/restart; it requires **systematic** investigation within existing permissions.

---

## Related

- **SYSTEMATIC_ERROR_TRACKING_PLAN.md** — checkpoints, error registry, ordered next steps, reporting template.
- **ERROR_TRACKING_STATUS.md** — rolling blocker notes and synthesis; update when registry changes.
- **CURRENT_STATE_AND_DIRECTION.md** — pipeline, blocker, and earlier permission wording (host: read logs only). This document **supersedes** the permission details there: host is now explicitly **read logs + read file contents**, **no editing**.
- **REFRESH_AND_GPU_DETECTION_INVESTIGATION.md** — refresh patch and rebuild/retest.
