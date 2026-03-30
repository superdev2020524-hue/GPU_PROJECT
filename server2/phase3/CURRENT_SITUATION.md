# Current situation

## The pipeline that was built and verified before

This is the path that had been made to work:

1. **Ollama** (in the guest VM) calls CUDA APIs.
2. **Guest shim** (libvgpu-cuda etc.) intercepts those calls and sends them to **VGPU-STUB**.
3. **VGPU-STUB** forwards the CUDA request to the **mediator**.
4. **Mediator** uses the **host’s CUDA** on the **physical GPU** to do the work and sends results back along the same chain.

That end-to-end path was implemented and verified; it’s what “working before” refers to.

---

## What is true right now

**On the VM:**

- **Ollama service** is running (active, not crashing).
- **Model list** works (API returns models like llama3.2:1b).
- **Model load / inference** fails: when you try to run a model you get errors like “error loading model: unexpectedly reached end of file” / “failed to load model”. So you cannot actually use a model.
- **Discovery** reports CPU and 0 B VRAM. So Ollama does not see a GPU.

**Why the pipeline is not working end-to-end:**

1. **Model loading is broken (with the shim loaded).**  
   The shim intercepts file open/read. Under that setup, the process that loads the model (e.g. reads the GGUF blob) fails: it can’t get a proper “real” libc file handle, and the fallback (syscall open/read) is not enough for the loader. So the model never loads, and inference never runs. This is the “failed to read magic” / “unexpectedly reached end of file” class of errors.

2. **The runner likely doesn’t use the shim.**  
   GPU discovery (e.g. “is there a GPU? how much VRAM?”) runs in a separate **runner** process. That runner is started as `ollama.real.bin runner ...`, not via a wrapper that sets `LD_PRELOAD`. So the runner probably does **not** load the vGPU shim. So it never intercepts CUDA, never talks to VGPU-STUB, and Ollama only sees CPU / 0 B VRAM.

So at the moment:

- The **server** process has the shim (LD_PRELOAD from systemd), but model loading fails there, so inference never gets to the point of using CUDA.
- The **runner** process (which would do discovery and likely the actual GPU work) probably does **not** have the shim, so the chain “Ollama → shim → VGPU-STUB → mediator → host GPU” is not in use for discovery or for compute.

---

## Summary in one sentence

**The shim/stub/mediator/host-GPU path exists and was working before; right now model loading fails when the shim is in use, the runner likely doesn’t load the shim so discovery shows CPU, and as a result the full “Ollama using the physical GPU via this pipeline” is not working.**
