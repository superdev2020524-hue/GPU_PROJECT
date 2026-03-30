# Upgrade Go on the VM and build Ollama there

*Mar 16, 2026*

## What “files required for that build” means

- **Ollama source:** Already on the VM at `/home/test-4/ollama/` with the vGPU patches (device.go, server.go, discover/runner.go) applied via `transfer_ollama_go_patches.py`. No need to move it again.
- **Go toolchain:** The VM has only Go 1.18; Ollama needs Go 1.23+. So the only thing to “move” or install is a **newer Go**:
  - **Option A:** Download the Go tarball **on the VM** from https://go.dev/dl/ (script does this by default).
  - **Option B:** Download the tarball on your PC, then **transfer** it to the VM with `--transfer-tarball` (use if the VM cannot reach go.dev or you prefer not to download on the VM).

## One-command upgrade and build

From the `phase3` directory:

```bash
python3 install_go_and_build_ollama_on_vm.py
```

This will:

1. **Download** Go 1.26.1 linux-amd64 on the VM from go.dev/dl (or use a tarball you provide).
2. **Extract** it to `/usr/local/go` (sudo).
3. **Build** `ollama.bin` in `/home/test-4/ollama` with `/usr/local/go/bin/go build -o ollama.bin .`
4. **Install** the new binary to `/usr/local/bin/ollama.bin` and restart the ollama service.

## If the VM cannot download (e.g. no outbound HTTPS)

1. On your PC (or any machine with a browser/curl), download:
   https://go.dev/dl/go1.26.1.linux-amd64.tar.gz

2. Run:

   ```bash
   python3 install_go_and_build_ollama_on_vm.py --transfer-tarball /path/to/go1.26.1.linux-amd64.tar.gz
   ```

   The script will transfer the tarball in chunks to the VM, then extract and build as above.

## If the tarball is already on the VM

```bash
python3 install_go_and_build_ollama_on_vm.py --tarball /tmp/go1.26.1.linux-amd64.tar.gz
```

## After a successful run

- `/usr/local/go/bin/go` will be Go 1.26.1 on the VM.
- Ollama will be running the **patched** binary (NeedsInitValidation skip for CUDA, runner env with `/opt/vgpu/lib`).
- Restart discovery by triggering a request or restarting ollama; the vGPU should no longer be filtered as “didn’t fully initialize” (see BOOTSTRAP_FIX_SKIP_CUDA_INIT_VALIDATION.md).

## Requirements

- `connect_vm.py`, `vm_config.py` (VM_USER, VM_HOST, VM_PASSWORD, REMOTE_HOME).
- VM has the patched Ollama tree at `REMOTE_HOME/ollama` (e.g. `/home/test-4/ollama`).
- VM has enough disk for the Go tarball (~64 MB) and the extracted tree (~200 MB under `/usr/local/go`), and for the Go build (module cache under `~/go` or in the ollama tree).
