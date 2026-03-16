# What to do on the host (GPU allocation fix)

**No files are overwritten in your repo.** Only the necessary minimal changes are already in `phase3/src/cuda_executor.c` (allocation and memory ops use primary context so test-4 does not hit "unable to allocate CUDA0 buffer").

---

## On the host (dom0 / GPU host)

Do these steps **on the machine where the mediator runs** (e.g. where you run `mediator_phase3` or `./mediator_phase3`).

### 1. Get the updated phase3 tree onto the host

Use one of:

- **SCP from your PC:**  
  From your PC (where this repo lives):  
  `scp -r /home/david/Downloads/gpu/phase3 <user>@<host>:/path/`  
  (Replace `<user>@<host>` and `/path/` with your host login and directory, e.g. `root@10.25.33.10:/root/`.)

- **Or git on the host:**  
  If the host has this repo cloned, run `git pull` (or checkout the branch that has the executor changes), then use that `phase3/` for the steps below.

### 2. Rebuild the mediator (so it uses the new executor)

On the host, in the **phase3** directory that contains the updated `src/cuda_executor.c`:

```bash
cd /path/to/phase3
make clean
make
```

This rebuilds `mediator_phase3` (and links in the updated `cuda_executor.o`).

- If the host does not have CUDA SDK / nvcc, the Makefile may still build the mediator if it only needs the executor; if the build fails, install the CUDA toolkit or use a build machine that has it and copy the built `mediator_phase3` binary to the host.

### 3. Restart the mediator

- If the mediator runs as a service (systemd):  
  `sudo systemctl restart <mediator-service-name>`

- If you run it by hand:  
  Stop the current process (Ctrl+C or kill), then start it again from the same phase3 directory:  
  `./mediator_phase3`  
  (or the same command you usually use, e.g. with a log file or config.)

### 4. Confirm

- Trigger a generate from the VM (test-4) and check mediator stderr or logs.
- You should see lines like:  
  `[cuda-executor] cuMemAlloc: allocating ... (vm=9)`  
  and  
  `[cuda-executor] cuMemAlloc SUCCESS: allocated 0x... (vm=9)`  
  instead of the guest error "unable to allocate CUDA0 buffer".

---

## Summary

| Step | Action |
|------|--------|
| 1 | Get updated phase3 onto the host (scp or git pull). |
| 2 | On host: `cd phase3 && make clean && make` |
| 3 | Restart the mediator (service or by hand). |
| 4 | Run a generate from test-4 and check mediator logs. |

No other changes are required on the host; no overwrites in your repo.
