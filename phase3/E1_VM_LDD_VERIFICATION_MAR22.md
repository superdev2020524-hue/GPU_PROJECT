# E1 — VM `ldd` verification (runner-equivalent path)

*2026-03-22 — next process after checklist: prove **libggml-cuda** pulls **cuda_v12** **libcublas / Lt** when **`LD_LIBRARY_PATH`** matches Ollama.*

## Command (VM)

```bash
export LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64
ldd /usr/local/lib/ollama/cuda_v12/libggml-cuda.so
```

## Result

| Library | Resolves to |
|---------|-------------|
| **`libcudart.so.12`** | **`/opt/vgpu/lib/libcudart.so.12`** (shim) |
| **`libcublas.so.12`** | **`/usr/local/lib/ollama/cuda_v12/libcublas.so.12`** → **12.3.2.9** |
| **`libcublasLt.so.12`** | **`/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12`** → **12.3.2.9** |
| **`libcuda.so.1`** | **`/opt/vgpu/lib/libcuda.so.1`** (shim) |
| **`libggml-base.so.0`** | **`/usr/local/lib/ollama/libggml-base.so.0`** |

**Note:** `ldd` may print *“no version information available”* for **`libcudart`** from the shim — usually harmless.

## Risk: multiple **`libcublasLt`** on disk

`find` shows **also** **`cuda_v13`** and **`mlx_cuda_v13`** **Lt** copies under **`/usr/local/lib/ollama/`**. If **`OLLAMA_LIBRARY_PATH`** or **`OLLAMA_LLM_LIBRARY`** ever point away from **`cuda_v12`**, the wrong **Lt** could load. Keep **`OLLAMA_LLM_LIBRARY=cuda_v12`** and **`cuda_v12` first** in **`OLLAMA_LIBRARY_PATH`**.

---

*This does not change **E1** root cause (**sm_80** in **`fail401312.bin`**); it confirms the **intended** DSO chain for the runner.*
