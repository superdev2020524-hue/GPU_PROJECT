# Phase 3 vs Your Requirements

Quick rundown of where Phase 3 matches what you asked for, and what we still need to add.

**TL;DR:** The core queuing and priority scheduling from Phase 2 works. Phase 3 adds rate limiting, better isolation, and more metrics. What's missing is explicit GPU scanning and the control panel view you described.

---

## What You Asked For vs What Phase 3 Does

### 1. Control Panel with GPU Scanning

**What you said:**
> "I imagine a control panel that scans the available hardware, identifies the physical GPUs in the system, and allows an administrator to assign virtual instances to those GPUs."

**Where we are:**
- Phase 3 expands the CLI (`vgpu-admin`) for managing VMs and pools
- Missing: GPU hardware scanning. We don't have a command to discover GPUs yet
- Missing: Explicit GPU-to-pool mapping. Pools exist but we don't show which GPU they map to

**What we need to add:**
- `vgpu-admin scan-gpus` - discover GPUs via nvidia-smi/lspci
- `vgpu-admin assign-gpu --gpu-id=0 --pool=A` - map GPU 0 to Pool A
- Make `vgpu-admin status` show GPUs, pools, and VMs together

This is straightforward to add. Just need to wire up nvidia-smi queries and store GPU info in the database.

---

### 2. VM-to-GPU Mapping (users 1-7 → GPU A, users 8-20 → GPU B)

**What you said:**
> "For example, virtual users 1 through 7 could be mapped to GPU A, while virtual users 8 through 20 are mapped to GPU B."

**Where we are:**
- Pool A and Pool B already exist with separate queues
- You can register VMs to pools: `vgpu-admin register-vm --pool=A --priority=high --vm-id=1`
- The architecture supports Pool A → GPU A, Pool B → GPU B

**What's missing:**
- The GPU-to-pool mapping isn't explicit. Right now pools are logical - we need to make the physical GPU assignment visible
- Need to show which GPU serves which pool in the status view

This is mostly a UI/database thing. The backend already routes Pool A requests to one GPU and Pool B to another. We just need to make that mapping explicit and visible.

---

### 3. Priority Levels (Low, Medium, High)

**What you said:**
> "It would also be valuable to assign priority levels—low, medium, or high—to each virtual instance so that more critical workloads are serviced first."

**Where we are:**
- This works. Phase 2 had basic priority, Phase 3 adds weighted priorities
- High priority requests go first, then medium, then low
- Within the same priority, it's FIFO
- We added aging so low-priority jobs don't starve forever
- Priority weights are configurable via CLI

**How it works:**
- Each VM gets a priority (0=low, 1=medium, 2=high) when registered
- Scheduler pulls from high-priority queue first, then medium, then low
- Same priority = FIFO order
- Aging bumps up old low-priority requests so they eventually run

**Status:** Done. This matches what you asked for.

---

### 4. Queuing System

**What you said:**
> "At its most basic level, this virtualization layer would function as a queuing system. Applications running within each virtual instance would issue function calls to the GPU. These calls would be queued, and when a call is ready to be serviced by its assigned physical GPU, control of the GPU would be handed to the requesting virtual instance. Once the function completes and the resulting data is returned to the virtual instance, the GPU would be released back to the queue for the next request."

**Where we are:**
- This is already working from Phase 2. The queuing mechanism is in place.

**How it works now:**
- VM apps call GPU functions via the vGPU stub device
- Mediation daemon queues requests (per-pool, sorted by priority)
- When a request is up, the daemon runs the CUDA job on the H100
- Results go back to the VM
- GPU is released, next request in queue runs

**What Phase 3 adds:**
- Rate limiting (token bucket) so VMs can't spam requests
- Queue depth limits per VM
- Watchdog to catch stuck jobs
- Better metrics: p95/p99 latency, queue depth, context switches

**Status:** Core queuing works. Phase 3 adds protection against abuse and better visibility.

---

### 5. Long-Running Function Protection

**What you said:**
> "I do see a potential challenge if a function call runs for an extended period, as this could block other users from accessing the GPU. For now, I think it makes sense to start with this simple model and then iterate from there. Over time, we can refine the system by introducing safeguards such as execution time limits, preemption, or scheduling timers to ensure that no single application monopolizes a physical GPU resource."

**Where we are:**
- Phase 2 uses the simple model: one job at a time per pool, execute, return, release
- Phase 3 adds safeguards without changing the execution model

**What Phase 3 adds:**
- Token bucket rate limiting - VMs can't submit requests faster than their rate limit
- Queue depth limits - a single VM can't fill the entire queue
- Watchdog monitoring - detects if a job runs too long (we log it, don't kill it yet)
- Quarantine mode - can temporarily disable a problematic VM

**What we're NOT doing yet:**
- Execution time limits (would need to kill jobs, more complex)
- Preemption (would need context switching, much more complex)
- Scheduling timers (similar complexity)

We're keeping the simple execution model you wanted, but adding protection so one VM can't starve others. If we need time limits or preemption later, we can add that in a future phase.

**Status:** Phase 3 prevents queue flooding and monitors long jobs, but keeps the simple execute-and-release model.

---

## Quick Summary

| What You Asked For | What We Have | Status |
|-------------------|--------------|--------|
| Control panel with GPU scanning | CLI exists, but no GPU scanning | Need to add GPU scan |
| GPU identification | Pools exist, but GPU mapping not explicit | Need to make mapping visible |
| VM-to-GPU mapping (1-7 → GPU A, 8-20 → GPU B) | Pool assignment works | Works, just needs better visibility |
| Priority levels (low/medium/high) | Weighted priority scheduling | Done |
| Queuing system | Per-VM queues, priority-ordered | Done |
| GPU handoff & release | Mediation daemon handles this | Done |
| Long-running function protection | Rate limiting + watchdog | Done |

---

## What We Need to Add to Phase 3

To match your control panel vision, we should add:

### 1. GPU Hardware Scanning
- `vgpu-admin scan-gpus` - query nvidia-smi to find GPUs
- Show GPU model, memory, PCI address, status
- Store in database so we can track which GPU is which

### 2. Explicit GPU-to-Pool Mapping
- `vgpu-admin assign-gpu --gpu-id=0 --pool=A` - map GPU 0 to Pool A
- Show which GPU serves which pool in status output
- Support multiple GPUs (GPU 0 → Pool A, GPU 1 → Pool B)

### 3. Better Status View
- Make `vgpu-admin status` show everything:
  - Physical GPUs and their status
  - Which GPU serves which pool
  - VMs in each pool
  - Priority breakdown
  - One screen = full system view

---

## Bottom Line

Phase 3 covers the core requirements:
- Priority scheduling (high/medium/low) - done
- Queuing with GPU handoff - done
- Isolation to prevent one VM from hogging the GPU - done
- Pool-based VM organization - done

What's missing: GPU scanning and making the GPU-to-pool mapping explicit. That's the gap between what we have and the control panel you described.

**Questions:**

1. **Control panel format:** CLI (what we have) or web dashboard? CLI is faster and works with CloudStack, but if you want a web UI we can build that.

2. **GPU scanning:** Manual (run `scan-gpus` when needed), automatic (on startup), or on-demand (when you view status)?

3. **Multi-GPU setup:** If you have multiple H100s, should each pool map to one GPU (1:1), or can multiple pools share one GPU (time-sliced)? Or both options?

Let me know what you think and we can adjust Phase 3 accordingly.
