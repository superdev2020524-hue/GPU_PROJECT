# VM inference verification (Mar 6)

## What was checked (connected to VM test-3)

1. **Generate result from earlier long-timeout run**
   - `/tmp/generate_result.json`: **not present** (or empty).
   - `/tmp/generate_curl.log`: present, **0 bytes**.
   - So the background generate we started did **not** complete with a written response (client or connection ended before the server could respond).

2. **Ollama and GPU path**
   - `ollama list`: **llama3.2:1b** (1.3 GB) is available.
   - Ollama logs show **cuMemcpyHtoD() SUCCESS** and **cudaMemcpy() SUCCESS** (1 MB and 256 KB chunks) — the guest is sending data to the host and the path is working.

3. **Host mediator (your log)**
   - VM 13 connected; **cuMemAlloc SUCCESS** (~1.3 GB × 2).
   - **HtoD progress: 10 MB … 140 MB total (vm=13)** — model load is progressing on the host GPU.
   - So: allocation works, data is flowing guest → host, but the **full model is ~1.3 GB**; at the rate you saw, load takes **well over 5 minutes**.

4. **Five-minute generate test**
   - A generate was run from the VM with a **5-minute** client timeout.
   - After 5 minutes, **no response file** was found (`/tmp/verify_result.json` missing).
   - So with a 5-minute timeout, the client does **not** get a successful response; the model is still loading.

## Conclusion

- **Pipeline:** Guest → shim → VGPU-STUB → mediator → physical GPU is **working** (allocations and HtoD copies succeed).
- **Issue:** Model load over this path is **slow** (many small round-trips). The **HTTP client** (e.g. curl) or the **Ollama server** gives up or the connection closes before the ~1.3 GB load finishes.
- **Fix in place:** `OLLAMA_LOAD_TIMEOUT=20m` in vgpu.conf so the **server** does not abort the load early. You still need a **client** that waits long enough (e.g. **15–20 minutes**).

## What you should do

1. **From the VM**, run a **single** generate with a **20-minute** client timeout and wait for it to finish:
   ```bash
   timeout 1200 curl -s -o /tmp/out.json -X POST http://localhost:11434/api/generate \
     -H "Content-Type: application/json" \
     -d '{"model":"llama3.2:1b","prompt":"Say hello.","stream":false}'
   ```
   Leave it running; watch the **mediator** for "HtoD progress" until it reaches ~1.3 GB (or stops increasing). When curl exits, check:
   ```bash
   ls -la /tmp/out.json
   head -c 300 /tmp/out.json
   ```

2. If **load completes** (mediator HtoD progress stops, runner goes to "Server listening") and curl returns **200** with JSON containing `"response"`, then inference is working and the only requirement is **long enough client and server timeouts**.

3. If the load **never** completes (e.g. mediator stalls, or Ollama logs "Load failed" again), then the next step is to capture the exact error and mediator/Ollama logs at that time.
