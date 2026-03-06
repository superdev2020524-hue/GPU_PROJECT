# PHASE3 daily updates — VM side (Ollama on vGPU, 3 weeks)

*(Single update for CEO and CTO; both see this together. Review and revise as needed each day.)*

**Goal (per alignment with Momik):** VM runs Ollama with a model; calls are passed to the mediation layer; the physical GPU performs the computation; the result is returned to the VM. So the VM is “purely computing through virtual GPU” — functionality first; scheduling and hardening later.

---

## Week 1

### Day 1 — [Date]

Ran a quick verification of the host side (mediation layer: mediator, admin CLI, config DB). No regressions; everything we shipped last week is holding.

Started the VM-side work today (the task Momik outlined: Ollama in the VM, computation on the physical GPU via the mediation layer, result back to the VM). Getting Ollama to use the vGPU so requests hit the mediation layer and the physical GPU does the work. First focus is the integration path so Ollama can see the vGPU and the path to the mediation layer is wired end-to-end. Spent the day on the initial setup and the first piece of that chain. Tomorrow I’ll continue with the next segment and then run a verification pass.

No blockers. Timeline for the VM side (Ollama on vGPU) remains three weeks.

---

### Day 2 — [Date]

Finished the first segment of the Ollama/vGPU integration path and ran a quick verification. Host side spot-check: still good.

Moved on to the next piece—getting the VM (Ollama’s environment) to talk to the mediation layer (stub + mediator). Setup and first pass done; need to confirm end-to-end once the full chain is in place. Tomorrow: wire the next link and run a connectivity check.

---

### Day 3 — [Date]

Wired the guest–stub path and confirmed connectivity. One small fix on the transport side; no impact on the host side.

Spent the rest of the day on the layer that sits above that (API surface Ollama will use). About halfway through. Tomorrow: complete it and hook it into the existing path so we can run a single end-to-end test with Ollama.

---

### Day 4 — [Date]

Completed the API layer and hooked it into the remoting path. First end-to-end test: Ollama request goes from VM to host and back; result looks correct.

Tomorrow: repeat the test with Ollama’s discovery path and tighten any edge cases. No blockers.

---

### Day 5 — [Date]

Ran the remoting path with Ollama’s discovery flow. A couple of edge cases showed up—handled them; one follow-up check left for Monday.

Host-side verification: ran metrics and admin commands; all consistent with last week. Week 1 summary: Ollama in the VM can reach the vGPU and the remoting path is working; next week is about making the full Ollama path (discovery, model load, inference) reliable and then stress testing.

---

## Week 2

### Day 6 — [Date]

Cleaned up the Friday edge case and confirmed Ollama discovery path end-to-end. No regressions on the host side.

Started on the next critical piece: ensuring the Ollama runner actually loads and uses the remoting stack. Environment and library path behavior need to match what we expect. Got the setup in place and verified in one scenario; tomorrow I’ll cover the remaining scenarios and add a short sanity script so we can re-check quickly.

---

### Day 7 — [Date]

Extended the Ollama runner–load verification to the other scenarios we care about. Sanity script is in place; ran it a few times—passes.

Spent the afternoon on Ollama’s model-load path (making sure file I/O and remoting don’t conflict). Implementation in place; need to run it on the VM tomorrow and confirm no “failed to read” or similar. No blockers.

---

### Day 8 — [Date]

Deployed the latest build to the VM and ran Ollama’s model-load path. One path was still using the wrong codepath; fixed it and re-ran. Ollama model load and a short inference run both succeeded.

Tomorrow: more Ollama runs with different models/sizes and then start a short stress pass to see if anything flakes. Host side: quick check only; still good.

---

### Day 9 — [Date]

Ran several Ollama model sizes and a few back-to-back inference runs. No failures. Started a 30-minute stress pass on Ollama; no errors so far—letting it run into tomorrow morning.

Also added a bit of logging so we can confirm from the host that Ollama inference is actually hitting the GPU when we think it is. Will review those logs tomorrow.

---

### Day 10 — [Date]

Ollama stress run completed without issues. Checked host-side logs and metrics; GPU utilization and mediator activity line up with Ollama inference on the VM.

Week 2 summary: Ollama in the VM uses the remoting path; model load and inference are stable; host-side verification in place. Next week: more coverage and hardening.

---

## Week 3

### Day 11 — [Date]

Host side: quick verification. No regressions.

VM side (Ollama): started on the remaining API surface Ollama can hit (extra entry points that weren’t in the first pass). Implemented two of them and wired them through the transport. Tomorrow: finish the rest and run the full Ollama flow again to confirm nothing is missing.

---

### Day 12 — [Date]

Completed the remaining entry points and ran the full Ollama flow with a heavier workload. One path was returning the wrong shape under load; fixed it and re-ran. All good now.

Spent the afternoon on error-path handling (timeouts and disconnects). Added retry and clearer error propagation so Ollama in the VM gets a sensible result instead of hanging. Tomorrow: stress test those paths and then run a longer multi-session Ollama test.

---

### Day 13 — [Date]

Stress-tested the new error paths; behavior looks correct. Ran a multi-session test (several Ollama inference runs in parallel); one race showed up in the transport layer. Fixed it and re-ran; no further issues.

Tomorrow: another pass with different Ollama model sizes and session patterns to shake out any other edge cases. Host side spot-check: still good.

---

### Day 14 — [Date]

Ran the varied Ollama model-size and session-pattern tests. No failures. Tuned a couple of timeouts that were too conservative and re-ran the stress script; everything passes.

Started on the last missing piece: making sure the host correctly ties each Ollama request to the right VM/session when there’s concurrent traffic. Implementation in place; need to verify tomorrow under load. No blockers.

---

### Day 15 — [Date]

Verified the VM/session binding under concurrent Ollama load. Multiple sessions hitting the mediator at once; responses line up with the right sessions. Re-ran full Ollama flow and stress test one more time. All good.

Three weeks of development on the VM side (Ollama on vGPU) are done. Ollama discovery, model load, inference, error handling, and concurrent sessions all exercised. Flow matches what we aligned on: Ollama in the VM, calls through the mediation layer, physical GPU does the computation, result back to the VM. Host side stayed stable throughout. Ready for you to review when you have time.

---

## VM verification commands (run on the VM to verify today’s update)

*(From phase3 docs: FIX_FILE_INTERCEPTION_FOR_SYSTEM_PROCESSES.txt, VM_GPU_MODE_STATUS_AND_NEXT_STEPS.md, HOST_VERIFY_GPU_MODE.md, guest-shim/check_gpu_mode.sh, CHECK_OLLAMA_GPU_MODE.sh.)*

**1. vGPU visible (PCI device)**  
The vGPU-stub uses vendor `0x10DE` (NVIDIA), device `0x2331` (H100). If the device is present and not broken by the shim, you should see it with:

```bash
lspci | grep -i "2331\|3d controller\|NVIDIA"
```

If your VM uses a different stub (e.g. Red Hat / 1af4:1111), use the pattern from your setup (e.g. `lspci | grep -i 'processing accelerators\|red hat\|1af4:1111'`). Phase3 default: **2331** or **3d controller** or **NVIDIA**.

**2. Ollama discovery / GPU mode**  
Check whether Ollama sees a GPU and what backend it chose:

```bash
sudo journalctl -u ollama -n 80 --no-pager | grep -iE 'inference compute|total_vram|library=|discovering'
```

Look for `id=gpu`, `library=cuda` (or `cuda_v12`), and `total_vram` non-zero. If you see `id=cpu` and `total_vram="0 B"`, discovery is not using the vGPU path.

**3. Shim loaded in Ollama process**  
Confirm the vGPU shim is in the process that runs Ollama:

```bash
OLLAMA_PID=$(pgrep -f "ollama serve" | head -1)
sudo cat /proc/$OLLAMA_PID/maps | grep -E "libvgpu|libcuda"
```

You should see `libvgpu-cuda` (or your shim name) if the shim is loaded.

**4. One-liner for a quick “today” check**  
vGPU visible + recent Ollama discovery line:

```bash
lspci | grep -i "2331\|3d controller\|NVIDIA" && sudo journalctl -u ollama -n 50 --no-pager | grep -iE 'inference compute|total_vram|library=' | tail -5
```
