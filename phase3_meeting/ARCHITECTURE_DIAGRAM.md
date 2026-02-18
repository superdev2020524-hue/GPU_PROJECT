# Architecture Diagrams: vGPU System

This document provides visual representations of the current and proposed architectures.

---

## Current Architecture: Custom Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                         GUEST VM                                │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │  Application     │                                           │
│  │  (Custom Client) │                                           │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           │ MMIO Write (Custom Protocol)                        │
│           │ Vendor ID: 0x1AF4                                   │
│           │ Device ID: 0x1111                                   │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  vGPU-Stub       │                                           │
│  │  (QEMU PCI Dev)  │                                           │
│  │  - MMIO BAR0     │                                           │
│  │  - Custom Regs   │                                           │
│  └────────┬─────────┘                                           │
└───────────┼─────────────────────────────────────────────────────┘
            │
            │ Unix Domain Socket
            │ (IPC)
            │
┌───────────┼─────────────────────────────────────────────────────┐
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  Mediator Daemon │                                           │
│  │  (Dom0 Host)     │                                           │
│  │                  │                                           │
│  │  ┌────────────┐  │                                           │
│  │  │  WFQ       │  │                                           │
│  │  │  Scheduler │  │                                           │
│  │  └─────┬──────┘  │                                           │
│  │        │         │                                           │
│  │  ┌─────▼──────┐  │                                           │
│  │  │  Rate      │  │                                           │
│  │  │  Limiter   │  │                                           │
│  │  └─────┬──────┘  │                                           │
│  │        │         │                                           │
│  │  ┌─────▼──────┐  │                                           │
│  │  │  Watchdog  │  │                                           │
│  │  └─────┬──────┘  │                                           │
│  │        │         │                                           │
│  │  ┌─────▼──────┐  │                                           │
│  │  │  CUDA      │  │                                           │
│  │  │  Executor  │  │                                           │
│  │  └─────┬──────┘  │                                           │
│  └────────┼─────────┘                                           │
│           │                                                     │
│           │ CUDA Runtime API                                    │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  NVIDIA H100     │                                           │
│  │  80GB PCIe       │                                           │
│  └──────────────────┘                                           │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │  SQLite DB       │                                           │
│  │  - VM Config     │                                           │
│  │  - Pools         │                                           │
│  │  - Metrics       │                                           │
│  └──────────────────┘                                           │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │  Admin CLI       │                                           │
│  │  (vgpu-admin)    │                                           │
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Custom vendor ID (0x1AF4) - not recognized by NVIDIA driver
- Custom MMIO register layout
- Custom protocol format
- VMs use custom client application
- Full control over scheduling and protocol

---

## Proposed Architecture: NVIDIA Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                         GUEST VM                                │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │  TensorFlow /    │                                           │
│  │  PyTorch /       │                                           │
│  │  CUDA Apps       │                                           │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           │ CUDA Runtime API                                    │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  NVIDIA Driver   │                                           │
│  │  (Auto-loaded)   │                                           │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           │ MMIO Write (NVIDIA Protocol)                        │
│           │ Vendor ID: 0x10DE (NVIDIA)                          │
│           │ Device ID: 0x2330 (H100)                            │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  vGPU-Stub       │                                           │
│  │  (QEMU PCI Dev)  │                                           │
│  │  - NVIDIA Regs   │                                           │
│  │  - NVIDIA Cmds   │                                           │
│  └────────┬─────────┘                                           │
└───────────┼─────────────────────────────────────────────────────┘
            │
            │ Unix Domain Socket
            │ (IPC) - SAME AS CURRENT
            │
┌───────────┼─────────────────────────────────────────────────────┐
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  Mediator Daemon │                                           │
│  │  (Dom0 Host)     │                                           │
│  │                  │                                           │
│  │  ┌────────────┐  │                                           │
│  │  │  NVIDIA    │  │                                           │
│  │  │  Parser    │  │  ← NEW: Parse NVIDIA commands             │
│  │  └─────┬──────┘  │                                           │
│  │        │         │                                           │
│  │  ┌─────▼──────┐  │                                           │
│  │  │  WFQ       │  │  ← SAME: Scheduler unchanged              │
│  │  │  Scheduler │  │                                           │
│  │  └─────┬──────┘  │                                           │
│  │        │         │                                           │
│  │  ┌─────▼──────┐  │                                           │
│  │  │  Rate      │  │  ← SAME: Rate limiter unchanged           │
│  │  │  Limiter   │  │                                           │
│  │  └─────┬──────┘  │                                           │
│  │        │         │                                           │
│  │  ┌─────▼──────┐  │                                           │
│  │  │  Watchdog  │  │  ← SAME: Watchdog unchanged               │
│  │  └─────┬──────┘  │                                           │
│  │        │         │                                           │
│  │  ┌─────▼──────┐  │                                           │
│  │  │  CUDA      │  │  ← SAME: CUDA execution unchanged         │
│  │  │  Executor  │  │                                           │
│  │  └─────┬──────┘  │                                           │
│  └────────┼─────────┘                                           │
│           │                                                     │
│           │ CUDA Runtime API                                    │
│           ▼                                                     │
│  ┌──────────────────┐                                           │
│  │  NVIDIA H100     │                                           │
│  │  80GB PCIe       │                                           │
│  └──────────────────┘                                           │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │  SQLite DB       │  ← SAME: Database unchanged               │
│  └──────────────────┘                                           │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │  Admin CLI       │  ← SAME: Admin CLI unchanged              │
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- NVIDIA vendor ID (0x10DE) - recognized by NVIDIA driver
- NVIDIA register layout (must implement)
- NVIDIA command format (must parse)
- VMs run standard CUDA applications
- TensorFlow/PyTorch work directly

**What Changes:**
- vGPU-stub: Complete rewrite of MMIO handlers
- Mediator: Command parser changes
- Guest: No custom client needed

**What Stays Same:**
- Unix socket communication
- Scheduler, rate limiter, watchdog
- CUDA execution
- Database and admin CLI

---

## Communication Flow: Current System

```
┌─────────┐
│   VM    │
│ Client  │
└────┬────┘
     │
     │ 1. MMIO Write (Custom Protocol)
     │    - Doorbell register
     │    - Request buffer (1KB)
     │
     ▼
┌─────────────┐
│  vGPU-Stub  │
│  (QEMU)     │
└──────┬──────┘
       │
       │ 2. Unix Socket IPC
       │    - /tmp/vgpu-mediator.sock
       │    - Binary protocol
       │
       ▼
┌─────────────┐
│  Mediator   │
│  Daemon     │
└──────┬──────┘
       │
       │ 3. WFQ Scheduler
       │    - Priority check
       │    - Rate limit check
       │    - Quarantine check
       │
       │ 4. CUDA Execution
       │    - cudaVectorAdd()
       │    - Async callback
       │
       │ 5. Response
       │    - Unix socket
       │    - MMIO read
       │
       ▼
┌─────────┐
│   VM    │
│ Client  │
└─────────┘
```

**Latency Breakdown:**
- MMIO write: 0.1-0.5ms
- Unix socket IPC: 0.1-0.5ms
- Scheduling decision: <0.001ms
- CUDA execution: 1-5ms
- **Total: 2-22ms typical**

---

## Multi-VM Scheduling Flow

```
┌─────────┐  ┌─────────┐  ┌─────────┐
│   VM1   │  │   VM2   │  │   VM3   │
│ (High)  │  │ (Med)   │  │  (Low)  │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     │            │            │
     └────────────┼────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  WFQ Scheduler  │
         │                 │
         │  Urgency Calc:  │
         │  - Priority     │
         │  - Weight       │
         │  - Queue Depth  │
         │  - Wait Time    │
         └────────┬────────┘
                  │
                  │ Highest Urgency First
                  ▼
         ┌─────────────────┐
         │  Rate Limiter   │
         │  (Token Bucket) │
         └────────┬────────┘
                  │
                  │ If Allowed
                  ▼
         ┌─────────────────┐
         │  Watchdog       │
         │  (Timeout Check)│
         └────────┬────────┘
                  │
                  │ If Not Quarantined
                  ▼
         ┌─────────────────┐
         │  CUDA Executor  │
         │  (H100 GPU)     │
         └─────────────────┘
```

**Scheduling Example:**
- VM1 (High priority, weight=80, queue=5) → Urgency: 4.0 × 1.6 × 1.5 = 9.6
- VM2 (Medium priority, weight=50, queue=3) → Urgency: 2.0 × 1.0 × 1.3 = 2.6
- VM3 (Low priority, weight=30, queue=1) → Urgency: 1.0 × 0.6 × 1.1 = 0.66

**Result:** VM1's request dequeued first

---

## Phase 4: CloudStack Integration Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    CloudStack UI                            │
│                                                             │
│  User: "Create VM with GPU"                                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              CloudStack Management Server                   │
│                                                             │
│  1. Query Host Agent: "Available vGPU capacity?"            │
│  2. Host Agent responds: "10 vGPUs available"               │
│  3. Create VM with vGPU assignment                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              XCP-ng Host (Dom0)                             │
│                                                             │
│  ┌──────────────────┐                                       │
│  │  Host GPU Agent  │  ← NEW: CloudStack API                │
│  │  (REST/XAPI)     │                                       │
│  └────────┬─────────┘                                       │
│           │                                                 │
│           │ Create vGPU Assignment                          │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  Mediator        │                                       │
│  │  (Existing)      │                                       │
│  └────────┬─────────┘                                       │
│           │                                                 │
│           │ Register VM in DB                               │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  SQLite DB       │                                       │
│  └──────────────────┘                                       │
│                                                             │
│  ┌──────────────────┐                                       │
│  │  QEMU            │                                       │
│  │  (vGPU-stub)     │                                       │
│  └──────────────────┘                                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Guest VM                                 │
│                                                             │
│  VM boots with vGPU-stub device attached                    │
│  VM can submit GPU requests                                 │
└─────────────────────────────────────────────────────────────┘
```

**Integration Points:**
1. Host GPU Agent API (REST or XAPI extension)
2. CloudStack plugin (detects vGPU capacity)
3. VM lifecycle hooks (start/stop → create/destroy vGPU)
4. UI/CLI support (GPU-enabled VM templates)

---

## Component Dependencies

```
┌─────────────────┐
│   Phase 1-2     │  ← IMPLEMENTED
│   (Foundation)  │
│   Phase 3       │  ← REMAINING WORK
│   (In Progress) │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│   Phase 4       │  │   Phase 5       │
│   CloudStack    │  │   Hardening     │
│   Integration   │  │   & Testing     │
└─────────────────┘  └─────────────────┘
         │                 │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │   Production    │
         │   Deployment    │
         └─────────────────┘
```

**Dependencies:**
- Phase 3 requires: Phase 1-2 implemented
- Phase 4 requires: Phase 1-3 done
- Phase 5 requires: Phase 1-3 done
- Production requires: Phase 3-5 done

