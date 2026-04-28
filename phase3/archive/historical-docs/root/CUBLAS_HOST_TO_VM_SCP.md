# Copy host cuBLAS → VM (paths + `scp`)

**Discovered on host (`10.25.33.10`):** `readlink -f` from `/usr/local/cuda/lib64/libcublas.so.12` resolves to:

| File | Host path |
|------|-----------|
| **libcublas** | `/mnt/cuda_install/cuda-12.3/targets/x86_64-linux/lib/libcublas.so.12.3.2.9` |
| **libcublasLt** | `/mnt/cuda_install/cuda-12.3/targets/x86_64-linux/lib/libcublasLt.so.12.3.2.9` |

**VM target directory (Ollama):** `/usr/local/lib/ollama/cuda_v12/`

**Version note:** The VM previously used **12.8.5.7** symlinks. These host files are **12.3.2.9** — you are **aligning to the host’s CUDA 12.3 install**, not keeping 12.8.5.7. That is intentional if the goal is “same bits as host.”

---

## 1) Run **on the host** (as a user that can read those paths, e.g. `root`)

Copy both files into the VM user’s home (or `/tmp`), then install with `sudo` **on the VM** (Step 2).

```bash
scp \
  /mnt/cuda_install/cuda-12.3/targets/x86_64-linux/lib/libcublas.so.12.3.2.9 \
  /mnt/cuda_install/cuda-12.3/targets/x86_64-linux/lib/libcublasLt.so.12.3.2.9 \
  test-4@10.25.33.12:/home/test-4/
```

(If `test-4` home is wrong on your VM, use e.g. `test-4@10.25.33.12:/tmp/` instead.)

---

## 2) Run **on the VM** (install + symlinks)

```bash
sudo mv /home/test-4/libcublas.so.12.3.2.9 /home/test-4/libcublasLt.so.12.3.2.9 /usr/local/lib/ollama/cuda_v12/
sudo ln -sf libcublas.so.12.3.2.9 /usr/local/lib/ollama/cuda_v12/libcublas.so.12
sudo ln -sf libcublasLt.so.12.3.2.9 /usr/local/lib/ollama/cuda_v12/libcublasLt.so.12
sudo systemctl restart ollama
```

**Verify:**

```bash
readlink -f /usr/local/lib/ollama/cuda_v12/libcublas.so.12
readlink -f /usr/local/lib/ollama/cuda_v12/libcublasLt.so.12
```

---

## 3) After you paste logs

Assistant can re-run VM checks (`ldd` with runner `LD_LIBRARY_PATH`, optional load + `mediator.log` grep for `401312`).
