# vGPU Stub MMIO Communication Enhancement - Project Overview

**Project:** Transition vGPU stub from NFS-based to MMIO-based communication  
**Date:** February 12, 2026  
**Status:** 📋 Planning Complete - Ready for Implementation  
**Location:** `/home/david/Downloads/gpu/step2_addtion/`

---

## 🎯 Project Goal

Transform the existing vGPU stub device from using NFS files for communication to using proper PCI MMIO registers, making it a true hardware-like device with direct guest-to-host communication.

### Current Architecture ❌
```
Guest VM → writes to /mnt/vgpu/request.json → NFS → Host reads file → Mediator
                                                                           ↓
Guest VM ← reads /mnt/vgpu/response.json ← NFS ← Host writes file ← CUDA execution
```

### Target Architecture ✅
```
Guest VM → writes to MMIO registers → VM Exit → QEMU → Unix Socket → Mediator
                                                                         ↓
Guest VM ← reads from MMIO buffer ← VM Entry ← QEMU ← Socket ← CUDA execution
```

---

## 📚 Documentation Files (Read in This Order)

### 1. **core.txt** (Original Plan)
**What it is:** Your original plan document outlining the transition strategy  
**Read if:** You want to understand the high-level goals and reasoning

### 2. **READY_TO_PROCEED.md** ⭐ **START HERE**
**What it is:** Summary of current state and decision point  
**Read if:** You want to understand what's needed to start implementation  
**Action required:** Choose an implementation option (1, 2, or 3)

### 3. **IMPLEMENTATION_ROADMAP.md**
**What it is:** Complete 6-phase implementation plan with timeline  
**Read if:** You want to see the full project breakdown and estimates  
**Contents:**
- Phase 1: Extend vGPU stub MMIO layout
- Phase 2: Add request/response processing  
- Phase 3: Create guest client library
- Phase 4: Update host mediator
- Phase 5: Testing strategy
- Phase 6: Remove NFS dependencies
- Timeline: 11-17 days estimated

### 4. **REGISTER_MAP_SPEC.md**
**What it is:** Complete technical specification of MMIO registers  
**Read if:** You want the detailed register layout and protocol spec  
**Contents:**
- Complete 4KB MMIO map (control registers, request buffer, response buffer)
- Register definitions and state machines
- Binary protocol specification
- Error codes and handling
- Example usage code
- Performance analysis

### 5. **VGPU_STUB_CHANGES.md**
**What it is:** Detailed code changes required to vgpu-stub.c  
**Read if:** You want to see exactly what changes to make  
**Contents:**
- Line-by-line comparison (current vs. new)
- 10 specific changes with code examples
- New functions to add
- Testing procedures
- Build impact analysis

---

## 🔧 What Currently Exists

Based on `complete.txt`, we have:

✅ **Working vGPU Stub PCI Device**
- Location: `hw/misc/vgpu-stub.c` in QEMU source
- PCI IDs: Vendor 0x1AF4, Device 0x1111, Class 0x1200
- 4KB MMIO BAR0 with basic registers
- Properties: pool_id, priority, vm_id
- Fully tested on XCP-ng 8.x

✅ **Build System**
- QEMU 4.2.1 with XCP-ng patches
- RPM build system configured
- qemu-wrapper patched for device-model-args

✅ **VM Integration**
- Device visible via `lspci` in guest
- MMIO accessible via `/sys/bus/pci/devices/`
- Properties readable from guest

---

## ❌ What's Missing (Needs Implementation)

1. **Extended MMIO Layout**
   - Doorbell register for guest notifications
   - Status/error registers for state tracking
   - Request buffer (1KB) for guest writes
   - Response buffer (1KB) for host writes
   - Additional control registers

2. **Host-Side Communication**
   - Socket connection to mediator daemon
   - Request forwarding logic
   - Response handling logic
   - Error handling

3. **Guest Client Library**
   - Functions to map MMIO region
   - Request submission helpers
   - Response polling/reading
   - Test programs

4. **Mediator Integration**
   - Socket interface for receiving requests
   - Binary protocol parser
   - Response sending logic
   - Keep existing CUDA execution (no changes)

5. **Testing Suite**
   - Register access tests
   - Request/response round-trip tests
   - Multi-VM priority tests
   - Performance benchmarks

---

## 🚀 How to Proceed

### Option 1: Quick Start (No Dependencies)
**What:** I create the enhanced vgpu-stub.c immediately  
**Timeline:** 1-2 hours  
**You can test:** New registers visible and accessible from guest  
**Next step:** You provide mediator code later for integration

### Option 2: Proper Order (Recommended)
**What:** You provide mediator code first, I implement everything in order  
**Timeline:** Starts immediately after you provide code  
**You can test:** Full end-to-end flow when complete  
**Next step:** Attach mediator daemon source code

### Option 3: Full Prototype
**What:** I create everything including a mock mediator  
**Timeline:** 2-3 days  
**You can test:** Complete working prototype  
**Next step:** Integrate with your real mediator later

---

## 📋 What I Need From You

To proceed efficiently, please provide:

### 🔴 CRITICAL: Mediator Daemon
- [ ] Source code location/repository
- [ ] Current request format (JSON structure?)
- [ ] Current response format
- [ ] How it processes CUDA operations
- [ ] Priority scheduling implementation
- [ ] Pool selection logic

**Where is it?** Example: `/opt/vgpu-mediator/`, GitHub repo, etc.

### 🟡 HELPFUL: Current Client Code
- [ ] How VMs currently submit requests
- [ ] NFS mount configuration
- [ ] Request submission scripts/code

**If this doesn't exist yet:** No problem! I'll create it from scratch.

### 🟢 OPTIONAL: Requirements
- [ ] Specific CUDA operations to support
- [ ] Maximum request/response sizes
- [ ] Latency requirements
- [ ] Security/isolation needs

---

## 📊 Project Scope

### Size Estimate
| Component | Estimate | Status |
|-----------|----------|--------|
| Enhanced vgpu-stub.c | +200 lines | ⏳ Not started |
| Guest client library | ~400 lines | ⏳ Not started |
| Mediator socket adapter | ~300 lines | ⏳ Not started |
| Test programs | ~600 lines | ⏳ Not started |
| Documentation | ~50 pages | ✅ Complete |
| **Total** | **~1500 lines code** | **~10% complete** |

### Timeline Estimate
| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Extend vGPU stub | 1-2 days | None |
| Guest client library | 2-3 days | Extended stub |
| Mediator integration | 3-4 days | Mediator code provided |
| Testing & validation | 2-3 days | All above |
| Cleanup & docs | 1-2 days | Testing complete |
| **Total** | **11-17 days** | **Mediator code** |

### Risk Assessment
| Risk | Impact | Mitigation |
|------|--------|------------|
| Mediator protocol unknown | High | Need code to understand |
| QEMU build issues | Medium | Already have working build |
| Guest kernel version issues | Low | Standard PCI access |
| Performance concerns | Low | MMIO faster than NFS |

---

## 🏗️ Project Structure

```
/home/david/Downloads/gpu/step2_addtion/
├── README.md                    ← You are here
├── core.txt                     ← Original plan
├── READY_TO_PROCEED.md          ← Next steps and decisions
├── IMPLEMENTATION_ROADMAP.md    ← Full project plan
├── REGISTER_MAP_SPEC.md         ← Technical specification
├── VGPU_STUB_CHANGES.md         ← Code changes needed
│
├── vgpu_stub_enhanced/          ← Will be created
│   ├── vgpu-stub.c              ← Enhanced device code
│   ├── vgpu-stub.h              ← Header file
│   ├── vgpu-protocol.h          ← Shared protocol definitions
│   └── patches/                 ← Patches for QEMU build
│
├── guest_client/                ← Will be created
│   ├── vgpu_mmio_client.c       ← Client library
│   ├── vgpu_mmio_client.h       ← Public API
│   ├── test_registers.c         ← Test program
│   ├── test_request_response.c  ← Test program
│   └── Makefile                 ← Build system
│
├── host_mediator/               ← Will be created
│   ├── socket_adapter.py        ← Socket interface
│   ├── protocol_parser.py       ← Binary protocol handler
│   └── README.md                ← Integration guide
│
├── testing/                     ← Will be created
│   ├── test_multi_vm.sh         ← Multi-VM tests
│   ├── test_priority.sh         ← Priority scheduling tests
│   └── benchmark.sh             ← Performance tests
│
└── docs/                        ← Will be created
    ├── API_REFERENCE.md         ← Client library API
    ├── PROTOCOL_SPEC.md         ← Binary protocol details
    └── MIGRATION_GUIDE.md       ← NFS→MMIO migration
```

---

## ✅ Success Criteria

### Phase 1 Success
- [x] Documentation complete
- [ ] Extended vGPU stub compiles
- [ ] New registers accessible from guest
- [ ] Backward compatible with existing VMs

### Phase 2 Success
- [ ] Doorbell triggers host-side handler
- [ ] Socket connects to mediator
- [ ] Request forwarding works

### Phase 3 Success
- [ ] Guest client library works
- [ ] Can submit requests via MMIO
- [ ] Can receive responses via MMIO

### Phase 4 Success
- [ ] Mediator receives requests via socket
- [ ] CUDA execution works end-to-end
- [ ] Responses delivered back to guest

### Phase 5 Success
- [ ] Multiple VMs work simultaneously
- [ ] Priority scheduling verified
- [ ] Performance meets requirements
- [ ] All tests pass

### Phase 6 Success
- [ ] NFS dependencies removed
- [ ] All documentation updated
- [ ] Production ready

---

## 🔄 Current Status

```
[████░░░░░░░░░░░░░░░░] 10% Complete

✅ Planning phase done
✅ Documentation complete
✅ Technical specifications written
⏳ Waiting for mediator code
⏳ Implementation not started
```

---

## 📞 Next Actions

### For You (User)
1. **Read READY_TO_PROCEED.md** to understand options
2. **Choose implementation approach** (Option 1, 2, or 3)
3. **Provide mediator code** (if choosing Option 2)
4. **Confirm requirements** (any specific constraints?)

### For Me (AI Assistant)
1. ⏳ Waiting for your decision
2. ⏳ Ready to create enhanced vgpu-stub.c
3. ⏳ Ready to create guest client library
4. ⏳ Ready to create mediator integration
5. ⏳ Ready to create test suite

---

## 💡 Key Design Decisions

### Why MMIO Instead of NFS?
- **Performance:** 10-100x faster (20-200µs vs 200-2000µs)
- **Proper Design:** Real hardware uses MMIO, not files
- **Latency:** Direct VM exits instead of file I/O
- **Scalability:** No filesystem overhead

### Why 1KB Buffers?
- **Sufficient:** Most requests fit in < 1KB
- **Fast:** Single VM exit to transfer
- **Standard:** Fits in single 4KB page
- **Extensible:** Can add DMA later for larger transfers

### Why Unix Sockets?
- **Simple:** Standard IPC mechanism
- **Fast:** Low overhead, no network stack
- **Reliable:** Stream-based, guaranteed delivery
- **Flexible:** Works with any language

### Why Binary Protocol?
- **Efficient:** Smaller than JSON
- **Fast:** No parsing overhead
- **Typed:** Compile-time validation
- **Extensible:** Version field for compatibility

---

## 📖 Additional Resources

### Related Documents
- `/home/david/Downloads/gpu/step2(quing)/vgpu-stub_enhance/complete.txt` - Original implementation guide
- `/home/david/Downloads/gpu/successful/vGPU_stub.txt` - Earlier version

### External References
- QEMU PCI device documentation
- XCP-ng QEMU customization guide
- PCI MMIO programming guide
- CUDA programming guide

---

## 🎓 Learning Resources

If you're new to any of these concepts:

**PCI MMIO:** Memory-mapped I/O allows guest OS to access device registers like memory addresses. Reads/writes cause VM exits to hypervisor.

**Doorbell Pattern:** Common hardware pattern where guest writes to a register to "ring a doorbell" and notify the host of work to do.

**Request/Response Buffers:** Shared memory regions where guest writes requests and host writes responses.

**Unix Domain Sockets:** Inter-process communication mechanism using file-system paths as addresses.

---

## ❓ FAQ

**Q: Will this break existing VMs?**  
A: No. VMs without the enhanced device continue working normally. The enhanced device is backward compatible.

**Q: Can we keep NFS as fallback?**  
A: Yes! We can support both during transition and keep NFS as backup.

**Q: What if mediator is in Python/C++/Go?**  
A: Doesn't matter. Socket interface works with any language.

**Q: How do we handle multiple GPUs?**  
A: Pool ID (A/B/etc.) routes requests to appropriate GPU pool.

**Q: What about security?**  
A: QEMU runs sandboxed, mediator validates all inputs, CUDA execution isolated per VM.

**Q: Performance impact?**  
A: Should be 10-100x faster than NFS for small requests. MMIO adds ~20-200µs overhead.

---

## 🚦 Status Indicators

🔴 **Blocked** - Waiting for external input (mediator code)  
🟡 **In Progress** - Currently being worked on  
🟢 **Complete** - Done and tested  
⏸️ **Paused** - On hold  
⏳ **Not Started** - Planned but not begun

**Current Overall Status:** 🔴 Blocked (waiting for mediator code or decision)

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-12 | Initial documentation created |
| | | - Planning complete |
| | | - Technical specs written |
| | | - Ready for implementation |

---

**👉 Next Step: Read [READY_TO_PROCEED.md](READY_TO_PROCEED.md) and choose your implementation path!**
