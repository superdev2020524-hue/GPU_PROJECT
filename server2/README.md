# Registry: `server2` — complete Phase 3 environment (second dom0)

This directory is a **full mirror** of the repository’s **`phase3/`** tree (minus large optional artifacts), packaged for the **second GPU / mediator host** and its workflows. It includes **mediator**, **`cuda_executor`**, **`vgpu-admin`**, **guest shims**, **VGPU stub** sources, **`Makefile`** (QEMU RPM targets), **`GOAL/`**, scripts, tests, and the **entire Phase 3 documentation history**.

---

## Layout

| Path | Role |
|------|------|
| **`phase3/`** | **Authoritative copy** — same layout as repo `phase3/`: `src/` (mediator, `cuda_executor.c`, **`vgpu-admin.c`**, **`vgpu-stub-enhanced.c`** (QEMU integration still names the device file `vgpu-stub.c` in the RPM), …), `include/`, `guest-shim/`, `Makefile`, `connect_host.py`, `connect_vm.py`, **`vm_config.py`** (in this registry: mediator **`10.25.33.20`**), deploy/transfer scripts, `tests/`, `GOAL/`, patches, and all `.md` / `.sh` work notes. |
| **`vm_config_server2.py`** | Optional defaults: mediator **`10.25.33.20`**. Copy to `phase3/vm_config.py` on a workstation if you always target server2 (or set `MEDIATOR_HOST` / `VM_*` via environment). |
| **`PHASE3_SYNC.md`** | `rsync` recipe used to refresh `phase3/` from the main repo. |
| **`CONNECTION.md`** | SSH / env notes for server2. |
| **`VERIFICATION_*.md`** | Dated evidence from dom0 checks (GPU, CUDA). |
| **`HOST2_DOM0_BRINGUP.md`** | Field notes from bringing up this dom0 (QEMU RPM, LFS, mediator, pitfalls). |
| **`HOST2_NEW_VM_SIMPLE.md`** | Short path: **wget ISO → XO create VM → `vgpu-admin register-vm`**. No bundled `create_vm.sh` here; full automation stays in repo root **`vm_create/`** if you need it on a matching layout. |

---

## What is included (vs excluded)

**Included:** Everything under `phase3/` needed to build and operate the stack: **vgpu-admin**, mediator, stub QEMU integration, guest shims, Python helpers, and docs.

**Excluded from mirror** (see `PHASE3_SYNC.md`): `out/`, `ollama-src/`, `ollama-src-phase3/`, `phase3_guest.tar`, `phase3.tar.gz`, `*.nohup.out`, nested `.git/`.

---

## First host vs server2

- **Primary dom0** in the main repo **`phase3/vm_config.py`** defaults to **`10.25.33.10`** — do not change that file for Host 2 work.
- **This registry** sets **`server2/phase3/vm_config.py`** to mediator **`10.25.33.20`** and the usual test VM password; run **`python3 connect_host.py '…'`** from **`server2/phase3/`** so imports resolve to that config. Override with **`MEDIATOR_HOST`** / **`MEDIATOR_PASSWORD`** if needed.
- **Server 2 VM rule:** for the guest-compute path, keep **XCP-ng host Secure Boot disabled** on the VM unless you are intentionally testing Secure Boot. The verified host command is:
  - **`xe vm-param-set uuid=<vm-uuid> platform:secureboot=false`**

---

## Isolation rule

For current **Server 2** work:

- edit and document only under **`server2/`**
- treat root **`phase3/`** as active **Server 1** work
- if you need to adapt a root `phase3` item, copy it into the **`server2`**
  registry before modifying it

See **`server2/phase3/SERVER2_ISOLATION_AND_MISSION_RULES.md`** and
**`server2/phase3/ASSISTANT_ROLE_AND_ANTICOUPLING.md`**.

---

## Assistant permissions

Unchanged: see **`phase3/ASSISTANT_PERMISSIONS.md`** inside the mirror. Host is read-only for the assistant unless you grant exceptions.
