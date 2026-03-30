# Host 2 — new VM without extra scripts

Use **Xen Orchestra** and **one directory** for ISOs. No `create_vm.sh` copy on this host unless you choose to use the main repo script later.

**Dom0:** pool master (e.g. `10.25.33.20`).

**Prerequisite:** **XO / orchestration** only works when the **XAPI toolstack and HTTPS path** on dom0 are healthy. If the **bundled UI** or **external XO** cannot talk to the host, fix **`yum`-level package alignment**, **DNS to mirrors**, and **`xe-toolstack-restart`** first (`server2/XCP_TOOLSTACK_AND_HTTPS_FIX.md`, **`HOST2_OPERATIONS_REPORT.md`** § Follow-up). Otherwise Step 2 below will fail no matter how good the ISO path is.

---

## Step 1 — ISO on disk

Pick the folder your **ISO library** SR uses in XO (example: `/mnt/cudawork/isos`). On dom0:

```bash
mkdir -p /mnt/cudawork/isos
cd /mnt/cudawork/isos
wget -c https://releases.ubuntu.com/22.04/ubuntu-22.04.5-live-server-amd64.iso
```

In **XO → Storage → your ISO SR → Rescan** so the ISO appears.

---

## Step 2 — Create the VM in XO

- **New VM** → template (e.g. Ubuntu) or **Other install media**.
- Set **CPU / RAM / disk / network**.
- Attach the **CD/DVD** = the ISO from Step 1.
- Start the VM, open **Console**, install the OS.

(If you prefer **only `xe`**, use the same flow you used before on the first server; the canonical scripted path in the repo is still **`vm_create/create_vm.sh`** at the **repository root**, not under `server2/`.)

---

## Step 3 — Phase 3 on this dom0

After the guest is installed and you know **`name-label`**:

```bash
vgpu-admin register-vm --vm-name="Your-VM-Name"
xe vm-start name-label="Your-VM-Name"   # if needed
```

Update **`server2/phase3/vm_config.py`** with the guest’s **IP / user / password** for workstation scripts.

---

That’s the full path: **ISO + XO + install + `vgpu-admin`**. No symlink helpers or Host-2-specific copies required unless XO/your SR layout needs the same fix as on pool 1 (then fix **once** in XO or follow **`vm_create/SOLUTION_SUMMARY.md`** in the main repo).
