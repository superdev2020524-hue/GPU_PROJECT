# NEXT STEPS - Implementation Guide
**Date:** 2026-02-08  
**Status:** Dom0 build complete ✅

---

## ✅ COMPLETED

1. ✅ All source code implemented
2. ✅ Dom0 build successful
   - `build-dom0/mediator_async` - MEDIATOR daemon
   - `build-dom0/cuda_vector_add` - CUDA test executable

---

## NEXT STEPS

### Step 1: Test CUDA Component (Optional but Recommended)

**On Dom0:**
```bash
cd /home/david/Downloads/gpu/step2_test
./build-dom0/cuda_vector_add
```

**Expected Output:**
```
CUDA Vector Addition Test
=======================

[CUDA] Initialized successfully (device count: 1)
[CUDA] Started async vector addition: 100 + 200
[TEST] Result: 300
Test 1: 100 + 200

[CUDA] Started async vector addition: 50 + 75
[TEST] Result: 125
Test 2: 50 + 75

All tests completed successfully!
```

**If this works:** CUDA is properly configured ✅  
**If this fails:** Check CUDA installation and GPU availability

---

### Step 2: Set Up NFS

**📖 See detailed guide:** `NFS_SETUP_GUIDE.md`

**Quick Setup:**

**On Dom0:**
```bash
# Install NFS server
yum install -y nfs-utils rpcbind

# Create directories
sudo mkdir -p /var/vgpu
sudo chmod 777 /var/vgpu
for i in {1..7}; do
    sudo mkdir -p /var/vgpu/vm$i
    sudo chmod 777 /var/vgpu/vm$i
done

# Configure export
echo '/var/vgpu *(rw,sync,no_root_squash,no_subtree_check,fsid=1,insecure)' | sudo tee -a /etc/exports

# Start services
sudo systemctl enable --now rpcbind nfs-server
sudo exportfs -rav
```

**On VM:**
```bash
# Install NFS client
sudo apt-get install -y nfs-common  # Ubuntu
# or
sudo yum install -y nfs-utils       # RHEL/CentOS

# Mount (replace with Dom0 IP)
sudo mkdir -p /mnt/vgpu
sudo mount -t nfs 10.25.33.10:/var/vgpu /mnt/vgpu
```

**For complete instructions, see `NFS_SETUP_GUIDE.md`**

---

### Step 3: Build VM Client on VM

**On VM (Test-1, Test-2, etc.):**

```bash
# Mount NFS (if not already mounted)
sudo mkdir -p /mnt/vgpu
sudo mount -t nfs <dom0-ip>:/var/vgpu /mnt/vgpu

# Copy source files to VM (or mount shared directory)
# Option 1: Copy files
scp user@dom0:/home/david/Downloads/gpu/step2_test/vm_client_vector.c /tmp/
scp user@dom0:/home/david/Downloads/gpu/step2_test/Makefile /tmp/

# Option 2: Mount shared directory (if available)

# Build VM client
cd /tmp
make vm

# Or build manually:
gcc -Wall -Wextra -O2 -g -o vm_client_vector vm_client_vector.c -lpthread
```

**Verify build:**
```bash
ls -la vm_client_vector
./vm_client_vector --help  # Should show usage
```

---

### Step 4: Configure VM vGPU Properties

**On Dom0, assign VMs to pools:**

```bash
# Using vgpu-admin (if installed)
vgpu-admin register-vm --vm-name="Test-1" --pool=A --priority=high --vm-id=1
vgpu-admin register-vm --vm-name="Test-2" --pool=A --priority=high --vm-id=2
vgpu-admin register-vm --vm-name="Test-3" --pool=A --priority=medium --vm-id=3
vgpu-admin register-vm --vm-name="Test-4" --pool=B --priority=high --vm-id=4
vgpu-admin register-vm --vm-name="Test-5" --pool=B --priority=medium --vm-id=5
vgpu-admin register-vm --vm-name="Test-6" --pool=B --priority=low --vm-id=6
vgpu-admin register-vm --vm-name="Test-7" --pool=A --priority=low --vm-id=7

# Or manually set device-model-args:
xe vm-param-set uuid=<vm-uuid> platform:device-model-args="-device vgpu-stub,pool_id=A,priority=2,vm_id=1"
```

**Verify vGPU device in VM:**
```bash
# Inside VM
lspci | grep "Processing accelerators"
# Should show the vGPU stub device
```

---

### Step 5: Start MEDIATOR Daemon

**On Dom0:**

```bash
cd /home/david/Downloads/gpu/step2_test
sudo ./build-dom0/mediator_async
```

**Expected Output:**
```
================================================================================
                    MEDIATOR DAEMON - CUDA Vector Addition
================================================================================

[MEDIATOR] Initialized
[CUDA] Initialized successfully (device count: 1)
[MEDIATOR] Starting main loop...
[MEDIATOR] Polling /var/vgpu every 1 seconds
```

**Keep this running** - it will process requests from VMs.

---

### Step 6: Test from VM

**On VM (e.g., Test-1):**

```bash
# Make sure NFS is mounted
mount | grep /mnt/vgpu

# Run VM client
sudo ./vm_client_vector 100 200
```

**Expected Output:**
```
================================================================================
                    VM CLIENT - Vector Addition Request
================================================================================

Request: 100 + 200

[MMIO] Read vGPU properties:
  Pool ID: A
  Priority: 2 (high)
  VM ID: 1

[REQUEST] Sent to MEDIATOR:
  Format: A:2:1:100:200
  File: /mnt/vgpu/vm1/request.txt

[WAIT] Polling for response...
  File: /mnt/vgpu/vm1/response.txt
  Timeout: 30 seconds
[RESPONSE] Received: 300

================================================================================
                    RESULT
================================================================================
  100 + 200 = 300
================================================================================
```

**On Dom0 (MEDIATOR output):**
```
[ENQUEUE] Pool A: vm=1, prio=2, 100+200
[PROCESS] Pool A: vm=1, prio=2, 100+200
[CUDA] Started async vector addition: 100 + 200
[RESULT] Pool A: vm=1, result=300
[RESPONSE] Sent to vm1: 300
[INIT] Cleared files for vm1
```

---

### Step 7: Test Priority Ordering

**Test Scenario: Multiple VMs with different priorities**

**On VM-1 (Pool A, High):**
```bash
sudo ./vm_client_vector 100 200
```

**On VM-4 (Pool B, High):**
```bash
sudo ./vm_client_vector 150 250
```

**On VM-2 (Pool A, Medium):**
```bash
sudo ./vm_client_vector 50 75
```

**Expected Processing Order:**
1. VM-1 (High priority, earlier)
2. VM-4 (High priority, later)
3. VM-2 (Medium priority)

---

### Step 8: Test Concurrent Requests

**Send requests from multiple VMs simultaneously:**

```bash
# On VM-1
sudo ./vm_client_vector 100 200 &

# On VM-4
sudo ./vm_client_vector 150 250 &

# On VM-2
sudo ./vm_client_vector 50 75 &
```

**Verify:**
- All requests are queued
- Processed in priority order
- All complete successfully

---

## TROUBLESHOOTING

### MEDIATOR not receiving requests
- Check NFS mount: `mount | grep /mnt/vgpu`
- Check file permissions: `ls -la /var/vgpu/vm*/`
- Check MEDIATOR is running: `ps aux | grep mediator_async`

### VM client can't read vGPU properties
- Verify vGPU stub is attached: `lspci | grep "Processing accelerators"`
- Check running as root: `sudo ./vm_client_vector ...`
- Verify MMIO path: `/sys/bus/pci/devices/0000:00:06.0/resource0`

### CUDA errors
- Check GPU: `nvidia-smi`
- Check CUDA: `nvcc --version`
- Test CUDA: `./build-dom0/cuda_vector_add`

### File permission issues
- Ensure directories are world-writable: `chmod 777 /var/vgpu/vm*`
- Check NFS export permissions

---

## SUCCESS CRITERIA

✅ MEDIATOR daemon running and processing requests  
✅ VMs can send requests and receive results  
✅ Priority ordering works correctly  
✅ Pool A and Pool B share same priority system  
✅ Files are initialized after response  
✅ Multiple VMs can submit concurrently  

---

## QUICK REFERENCE

**Dom0 Commands:**
```bash
# Start MEDIATOR
sudo ./build-dom0/mediator_async

# Test CUDA
./build-dom0/cuda_vector_add

# Check NFS
sudo exportfs -v
ls -la /var/vgpu/vm*/
```

**VM Commands:**
```bash
# Mount NFS
sudo mount -t nfs <dom0-ip>:/var/vgpu /mnt/vgpu

# Run client
sudo ./vm_client_vector <num1> <num2>

# Check vGPU
lspci | grep "Processing accelerators"
```

---

**Ready to proceed! Start with Step 1 (test CUDA) or Step 2 (set up NFS).**
