# Assistant permissions (Phase 3)

*Clarified: Mar 18, 2026 — after host/VM modifications for VGPU display name (lspci) and rebuild.*

*Host expanded: 2026-03-25 — operator granted full dom0 operational authority for PHASE3 work, with an explicit **non-destruction** condition (see **Host** below).*

---

## Host

### Current grant (2026-03-25)

The operator authorizes the assistant to act **autonomously on the mediator host (dom0)** for PHASE3-related work, including:

- **Read** logs and file contents (e.g. `/tmp/mediator.log`, build trees).
- **Edit** source and config under agreed project paths (e.g. `/root/phase3`).
- **Build** (`make`, `nvcc`, etc.) and **install** Phase 3 host binaries where documented.
- **Restart** mediator / related services after changes, when required for verification.
- **Deploy** from the workspace (e.g. `deploy_cuda_executor_to_host.py`, `connect_host.py`).

### Binding condition — do not destroy the host

All host actions must **preserve host integrity**:

- **No** destructive bulk deletion (e.g. `rm -rf` on `/`, `/boot`, `/etc`, `/usr`, vendor CUDA trees).
- **No** disk or filesystem operations intended to brick dom0 or wipe role-critical state.
- **Prefer** backing up a file before overwriting known-good configs or single critical paths.
- **Stay** within PHASE3 / mediator / documented deployment scope unless the operator directs otherwise.

If a change is unusually risky, **stop and ask** even under this grant.

The operator may **narrow or revoke** host authority at any time; **`ASSISTANT_PERMISSIONS.md`** should be updated when that happens.

### Historical note (before 2026-03-25)

Host was **read-only** for the assistant (logs + read files only); dom0 edits and rebuilds were documented for the operator to apply.

---

## VM (test-4)

- **Full authority:** Run commands, configure, deploy guest artifacts, edit VM files, read VM logs, rebuild and install software on the VM (e.g. ollama.bin, guest shims), restart services.

---

## Summary

| Scope   | Logs / read files | Edit / deploy / build / restart |
|---------|-------------------|----------------------------------|
| **Host** | Yes               | Yes — **within non-destruction condition above** |
| **VM**   | Yes               | Yes (full)                       |

---

## Role and anti-coupling

- **ASSISTANT_ROLE_AND_ANTICOUPLING.md** — On error: first search PHASE3 for past resolutions; verify no negative impact on previously working behavior (e.g. GPU mode, runner env); for VM build always check `/usr/local/go/bin/go version` before concluding the VM cannot build.
- **SYSTEMATIC_ERROR_TRACKING_PLAN.md** — **Mandatory** operational procedure: checkpoints **A–D**, error registry **E1/E2/…**, gates before long runs, and required fields in assistant status updates (see **ASSISTANT_ROLE_AND_ANTICOUPLING.md §5**).

---

## Assistant operational obligations (Phase 3 / Phase 1)

When triaging errors or advancing the Phase 1 milestone, the assistant **must**:

1. Follow **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`** and **`ASSISTANT_ROLE_AND_ANTICOUPLING.md` §5** (not ad-hoc “run and wait”). **Long-term model loading** (multi-hour / long single-load windows) requires **explicit operator approval** per **`ASSISTANT_ROLE_AND_ANTICOUPLING.md` §5.4**.
2. Use **`connect_vm.py`** and **`connect_host.py`** to capture **Checkpoint A–C** evidence before recommending long generates. Host **edits/builds/restarts** are allowed **only** within **`ASSISTANT_PERMISSIONS.md`** (including the **non-destruction** condition).
3. Record material changes in **`ERROR_TRACKING_STATUS.md`** (short dated note + registry state).

---

## Related

- **SYSTEMATIC_ERROR_TRACKING_PLAN.md** — checkpoints, error registry, ordered next steps, reporting template.
- **ERROR_TRACKING_STATUS.md** — rolling blocker notes and synthesis; update when registry changes.
- **CURRENT_STATE_AND_DIRECTION.md** — pipeline and blocker narrative; for **permission details**, this document (**ASSISTANT_PERMISSIONS.md**) is authoritative.
- **REFRESH_AND_GPU_DETECTION_INVESTIGATION.md** — refresh patch and rebuild/retest.
