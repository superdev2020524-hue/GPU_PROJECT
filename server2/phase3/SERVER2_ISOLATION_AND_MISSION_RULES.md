# Server 2 isolation and mission rules

*Added: Apr 17, 2026 - mandatory for current Server 2 work.*

---

## Purpose

This document prevents **Server 1** and **Server 2** work from being mixed.

For the current mission, the assistant must treat **`server2/phase3/`** as the
authoritative working tree for all Server 2 research, notes, scripts, edits,
and deployment preparation.

---

## Mandatory boundary

1. **Do not edit** the repository root **`phase3/`** tree for this Server 2
   mission.
2. Treat the repository root **`phase3/`** tree as **Server 1** work that may be
   active in parallel and must not be disrupted.
3. For Server 2, use only:
   - **`server2/`**
   - **`server2/phase3/`**
   - files newly created under the **`server2/`** tree
4. If a useful procedure or document exists only in the root **`phase3/`**
   history, **do not modify it there**. Instead:
   - read it,
   - copy or mirror the needed content into the **`server2`** registry if
     modification is required,
   - then continue from the **Server 2** copy.
5. Do not update root `phase3` status notes, role notes, permissions notes,
   helper scripts, or deployment files as part of this mission.

---

## Role registration

For this mission, the assistant is registered as:

- **Mission:** bring up a working **Server 2** path quickly and safely
- **Primary host:** `10.25.33.20`
- **Current target VM:** `10.25.33.21`
- **Primary registry:** **`server2/phase3/`**
- **Protected parallel track:** root **`phase3/`** for **Server 1**
- **Chosen final exposure method:** real GPU PCI passthrough for Server 2
- **Guest policy for the final path:** no mediated-path CUDA/NVML shims may remain active in the deployed passthrough VM

This registration remains in force unless the user explicitly changes it.

---

## Delivery goal

The Server 2 deliverable must optimize for the shortest path to a working
deployment:

1. `lspci` in the VM must show the **HEXACORE** presentation.
2. General GPU applications in the VM must function normally.
3. The final method must be easy to repeat on a **new VM** created later by the
   hosting administrator.
4. The host-side attach path should stay as simple as possible, ideally
   centered on the existing Server 2 flow such as:
   - create VM
   - run `server2/attach_passthrough_vm.sh <vm-uuid>` on the host
   - for a fresh VM, finish guest branding over SSH with `server2/phase3/fix_pci_ids_vm.py`
   - use `server2/phase3/clean_passthrough_vm.py` only for older mixed VMs that still contain mediated-path leftovers

---

## Operational rules

1. Reuse verified Phase 3 methods, but only from the **Server 2 registry** when
   editing or documenting.
2. Keep the existing Phase 3 discipline:
   - systematic error tracking,
   - permissions control,
   - anti-coupling verification,
   - no blind long runs.
3. Prefer the fastest validated method over architectural redesign when time is
   limited.
4. Before any change that could affect both tracks, confirm the change stays
   inside **`server2`** only.

---

## Related

- **`ASSISTANT_ROLE_AND_ANTICOUPLING.md`**
- **`ASSISTANT_PERMISSIONS.md`**
- **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`**
- **`../../README.md`**
