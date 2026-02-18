# Q&A Reference: Anticipated Client Questions

Prepared answers to questions the client may ask during the meeting.

---

## Progress & Status Questions

### Q: "What's the current status? What's working today?"

**Answer:**
- **Working:** Multi-VM GPU sharing with 2-4 VMs on single H100
- **Working:** Priority-based scheduling (high/medium/low)
- **Working:** Pool-based isolation (Pool A/B)
- **In Progress:** Weighted fair queuing scheduler (Phase 3)
- **In Progress:** Per-VM rate limiting and queue depth controls (Phase 3)
- **In Progress:** Automatic VM quarantine on faults (Phase 3)
- **In Progress:** Metrics collection (p50/p95/p99 latency, Prometheus export) (Phase 3)
- **In Progress:** GPU health monitoring (NVML integration) (Phase 3)
- **In Progress:** Admin CLI extensions for Phase 3 features
- **Pending:** CloudStack integration (Phase 4)
- **Pending:** Production hardening and large-scale testing (Phase 5)

---

### Q: "What's needed for production deployment?"

**Answer:**
- **Current:** Functional for testing and development with 2-4 VMs
- **For Production:** Need Phase 4 (CloudStack integration) + Phase 5 (hardening)
- **Remaining Work:** CloudStack plugin development, end-to-end testing with 15-30 VMs, security audit (IOMMU implementation)

---

### Q: "What's the maximum number of VMs per H100?"

**Answer:**
- **Tested:** 2-4 VMs (validated)
- **Target:** 15-30 VMs (Phase 5 goal, needs validation)
- **Limiting Factors:**
  - GPU memory per VM (80GB shared)
  - Context switching overhead
  - Queue depth and scheduling latency
  - Real-world workload characteristics
- **Recommendation:** Start with 10 VMs, scale up based on benchmarks

---

## Technical Architecture Questions

### Q: "How does the VM communicate with the GPU?"

**Answer:**
- **Current Architecture:**
  1. VM writes to MMIO registers in vGPU-stub PCI device (custom vendor ID 0x1AF4)
  2. vGPU-stub forwards request via Unix domain socket to mediator daemon
  3. Mediator schedules and executes on real H100 via CUDA
  4. Response flows back through same path
- **Custom Protocol:** VMs use custom client application (`vm_client_enhanced.c`)
- **Not Direct CUDA:** VMs do NOT run NVIDIA driver or CUDA directly

---

### Q: "Can TensorFlow or PyTorch run directly in VMs?"

**Answer:**
- **Current:** ❌ NO - Requires custom client application
- **Why:** vGPU-stub uses custom vendor ID, not recognized by NVIDIA driver
- **Future Option:** Pivot to NVIDIA protocol (vendor ID 0x10DE)

---

### Q: "What about security? Is this IOMMU-protected?"

**Answer:**
- **Current:** Using Unix domain sockets (secure within Dom0)
- **IOMMU Status:** Not yet implemented (was Phase 2 Step 1, deferred)
- **Security Level:** Suitable for development/testing, needs IOMMU for production
- **Risk:** Medium - Communication is host-internal, but not hardware-isolated
- **Recommendation:** Implement IOMMU protection before production deployment

---

## CloudStack Integration Questions

### Q: "How will this integrate with CloudStack?"

**Answer:**
- **Phase 4 Plan:**
  1. Host-side GPU Agent API (REST or XAPI extension)
  2. CloudStack plugin to detect vGPU capacity
  3. VM lifecycle hooks (start/stop → create/destroy vGPU assignment)
  4. UI/CLI support for GPU-enabled VM templates
- **Status:** Design phase, not implemented
- **Complexity:** Medium-High (requires CloudStack plugin development)
- **Scope:** Host-side GPU Agent API, CloudStack plugin, VM lifecycle integration, UI/CLI support

---

### Q: "Can CloudStack automatically assign vGPUs to VMs?"

**Answer:**
- **Future (Phase 4):** Yes, via CloudStack plugin
- **Current:** Manual assignment via `vgpu-admin` CLI
- **Workflow (Future):**
  - User requests VM with GPU in CloudStack UI
  - CloudStack queries host agent for available vGPU capacity
  - Host agent creates vGPU assignment via mediator
  - VM boots with vGPU-stub device attached

---

## Performance & Scalability Questions

### Q: "What's the performance overhead?"

**Answer:**
- **Latency:** 2-22ms typical (vs 500-1100ms with old NFS method)
- **Overhead Sources:**
  - MMIO write: 0.1-0.5ms
  - Unix socket IPC: 0.1-0.5ms
  - Scheduling decision: <0.001ms
  - CUDA execution: 1-5ms (same as native)
- **Context Switch Cost:** ~0.1ms per switch
- **Overall:** <5% overhead for typical workloads

---

### Q: "How does scheduling work? Is it fair?"

**Answer:**
- **Algorithm:** Weighted Fair Queuing (WFQ)
- **Factors:**
  - VM priority (high/medium/low) - 4x/2x/1x multiplier
  - Admin-assigned weight (1-100, default 50)
  - Queue depth pressure
  - Wait time (prevents starvation)
- **Fairness:** Yes - lower priority VMs guaranteed access, no starvation
- **Configurable:** Per-VM weights and rate limits via admin CLI

---

### Q: "What happens if a VM misbehaves?"

**Answer:**
- **Automatic Quarantine:** Watchdog monitors job timeouts (default 30s)
- **Fault Tracking:** Error counter per VM (threshold: 5 failures)
- **Auto-Isolation:** Quarantined VMs cannot submit new jobs
- **Manual Override:** Admin can quarantine/unquarantine via CLI
- **Recovery:** Admin clears quarantine flag when issue resolved

---

## Strategic Direction Questions

### Q: "Should we pivot to NVIDIA protocol now or later?"

**Answer:**
- **Custom Protocol:** Simpler implementation path, but requires custom clients
- **NVIDIA Protocol:** TensorFlow/PyTorch work directly, but requires significant implementation changes
- **Recommendation:** Hybrid approach - finish Phase 3 and Phase 4/5 with custom protocol, then evaluate NVIDIA protocol as Phase 6

---

### Q: "What's the difference between current approach and NVIDIA protocol?"

**Answer:**
- **Current (Custom Protocol):**
  - Custom vendor ID (0x1AF4), custom MMIO registers
  - VMs use custom client application
  - Full control over protocol and scheduling
  - Cannot run TensorFlow/PyTorch directly
  
- **NVIDIA Protocol (Proposed):**
  - NVIDIA vendor ID (0x10DE), NVIDIA register layout
  - NVIDIA driver loads automatically in VM
  - TensorFlow/PyTorch work without modification
  - Must implement NVIDIA's hardware interface (complex)

---

### Q: "Which approach is better for our use case?"

**Answer:**
- **Custom Protocol:** Simpler CloudStack integration, proven architecture, continue current implementation path
- **NVIDIA Protocol:** TensorFlow/PyTorch work directly, but requires significant implementation changes
- **Recommendation:** Finish Phase 4/5 with custom protocol, then evaluate NVIDIA protocol as Phase 6

---

## Questions to Ask Client

1. **Priority:** What's the primary use case - CloudStack integration OR TensorFlow/PyTorch support?
2. **Workloads:** What applications will run in VMs? (affects NVIDIA protocol decision)
3. **Scale:** How many VMs per H100 in production?
4. **Security:** Is IOMMU required before production, or can we deploy without it initially?
5. **Protocol Direction:** Should we continue with custom protocol or pivot to NVIDIA protocol?
