# VM connection (connect_vm.py)

## How it works

- **Config:** `vm_config.py` — `VM_USER`, `VM_HOST`, `VM_PASSWORD` (e.g. test_4@10.25.33.12).
- **Auth:** pexpect drives SSH and sends the password when prompted (`password:` or `Password:`).
- **Prompt:** Expects a shell prompt (`$` or `#`) or password prompt within `PROMPT_TIMEOUT_SEC` (25s).

## Why “Unexpected response” or “Connection timed out”?

The script does **not** change how SSH connects. It only automates password entry. If you see:

- **`SSH output/error: Connection timed out`** or **`No route to host`**  
  The **machine running the script** cannot reach the VM. The VM (e.g. 10.25.33.11) is likely on a private network (e.g. lab LAN or VPN). Cursor’s agent/cloud runs elsewhere and has no route to that IP.

- **`Unexpected response (no output from SSH)`**  
  SSH produced no output before the timeout (e.g. firewall, SSH not answering).

## What to do

1. **Run from a host that can reach the VM**  
   From your laptop or a jump host on the same network as the VM:
   ```bash
   cd /path/to/phase3
   python3 connect_vm.py "echo ok"
   ```
   If that works, the script and credentials are fine; the earlier failure was due to the execution environment (e.g. Cursor backend) not having network access to the VM.

2. **Check from that host:**
   ```bash
   ping -c 2 10.25.33.11
   ssh -o ConnectTimeout=5 test_4@10.25.33.12 "echo ok"   # will prompt for password
   ```

3. **Timeouts (optional):**  
   `CONNECT_VM_CONNECT_TIMEOUT_SEC`, `CONNECT_VM_PROMPT_TIMEOUT_SEC`, `CONNECT_VM_COMMAND_TIMEOUT_SEC` can be set in the environment if you need longer waits.

## Summary

- Connection succeeded earlier when the script was run from a machine that could reach the VM.
- Connection fails in environments (e.g. Cursor cloud) that have no route to 10.25.33.11.
- Run `connect_vm.py` from a host that can ping/SSH the VM (e.g. your workstation or a jump host on the same LAN/VPN).
