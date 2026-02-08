# VM Directory Mapping - How It Works
**Date:** 2026-02-08  
**Purpose:** Explain how VM-to-directory mapping works (NOT automatic!)

---

## IMPORTANT: This is NOT Automatic!

The mapping between VMs and directories is **manual** and requires **proper configuration**. Here's how it works:

---

## HOW THE MAPPING WORKS

### Step 1: Directory Creation (Manual on Dom0)

**On Dom0, you manually create:**
```
/var/vgpu/vm1/  → Intended for Test-1
/var/vgpu/vm2/  → Intended for Test-2
/var/vgpu/vm3/  → Intended for Test-3
...
```

**This is just directory structure - no automatic mapping yet!**

---

### Step 2: VM Configuration (Manual - You Must Configure Each VM)

**Each VM must be configured with the correct `vm_id` in its vGPU stub device:**

**Example for Test-1:**
```bash
# On Dom0, configure Test-1's vGPU stub
xe vm-param-set uuid=<test-1-uuid> \
  platform:device-model-args="-device vgpu-stub,pool_id=A,priority=2,vm_id=1"
```

**Example for Test-4:**
```bash
# On Dom0, configure Test-4's vGPU stub
xe vm-param-set uuid=<test-4-uuid> \
  platform:device-model-args="-device vgpu-stub,pool_id=B,priority=2,vm_id=4"
```

**The `vm_id` parameter is what determines which directory the VM will use!**

---

### Step 3: VM Client Reads vm_id (Automatic - Code Does This)

**When VM client runs, it reads the `vm_id` from vGPU MMIO:**

```c
// In vm_client_vector.c
props->vm_id = mmio[0x010/4];  // Read vm_id from MMIO register

// Then uses it to construct the path:
snprintf(request_file, sizeof(request_file), 
         "/mnt/vgpu/vm%u/request.txt", props->vm_id);
// If vm_id=1 → /mnt/vgpu/vm1/request.txt
// If vm_id=4 → /mnt/vgpu/vm4/request.txt
```

**So if Test-1 is configured with `vm_id=1`, it will automatically use `vm1/` directory.**

---

### Step 4: MEDIATOR Reads vm_id from Request (Automatic - Code Does This)

**MEDIATOR parses the request to get vm_id:**

```c
// Request format: "pool_id:priority:vm_id:num1:num2"
// Example: "A:2:1:100:200"  (Pool A, priority 2, vm_id=1, 100+200)

// MEDIATOR parses this and uses vm_id to write response:
snprintf(response_file, sizeof(response_file),
         "%s/vm%u/response.txt", NFS_BASE_DIR, vm_id);
// If vm_id=1 → /var/vgpu/vm1/response.txt
```

---

## THE MAPPING IS ENFORCED BY CODE LOGIC

### What's Automatic:
✅ VM client reads `vm_id` from vGPU MMIO  
✅ VM client uses `vm_id` to construct directory path  
✅ MEDIATOR uses `vm_id` from request to write response  

### What's Manual (You Must Do):
❌ Create directories on Dom0  
❌ Configure each VM with correct `vm_id` in vGPU stub  
❌ Ensure `vm_id` matches intended directory number  

---

## EXAMPLE WORKFLOW

### Setting Up Test-1:

**1. On Dom0 - Create directory:**
```bash
mkdir -p /var/vgpu/vm1
chmod 777 /var/vgpu/vm1
```

**2. On Dom0 - Configure Test-1 VM:**
```bash
# Get Test-1 UUID
TEST1_UUID=$(xe vm-list name-label="Test-1" params=uuid --minimal)

# Configure vGPU stub with vm_id=1
xe vm-param-set uuid=$TEST1_UUID \
  platform:device-model-args="-device vgpu-stub,pool_id=A,priority=2,vm_id=1"
```

**3. On Test-1 VM - VM client automatically uses vm1/:**
```bash
# VM client reads vm_id=1 from MMIO
# Automatically constructs: /mnt/vgpu/vm1/request.txt
sudo ./vm_client_vector 100 200
```

**4. MEDIATOR automatically writes to vm1/:**
```bash
# MEDIATOR parses request: "A:2:1:100:200"
# Extracts vm_id=1
# Automatically writes to: /var/vgpu/vm1/response.txt
```

---

## WHAT HAPPENS IF YOU MISCONFIGURE?

### Wrong vm_id Configuration:

**If Test-1 is configured with `vm_id=4` instead of `vm_id=1`:**

```bash
# Wrong configuration:
xe vm-param-set uuid=$TEST1_UUID \
  platform:device-model-args="-device vgpu-stub,...,vm_id=4"  # WRONG!

# Result:
# - Test-1 will try to use /mnt/vgpu/vm4/request.txt
# - MEDIATOR will write to /var/vgpu/vm4/response.txt
# - Test-4 might receive Test-1's responses (if Test-4 also uses vm_id=4)
# - **This causes confusion and errors!**
```

**Solution:** Always ensure `vm_id` matches the intended directory number!

---

## RECOMMENDED CONFIGURATION

### Use vgpu-admin (if available):

```bash
# Register Test-1 with vm_id=1
vgpu-admin register-vm --vm-name="Test-1" --pool=A --priority=high --vm-id=1

# Register Test-4 with vm_id=4
vgpu-admin register-vm --vm-name="Test-4" --pool=B --priority=high --vm-id=4
```

**This automatically:**
- Sets the correct `vm_id` in vGPU stub
- Ensures consistency
- Prevents configuration errors

---

## VERIFICATION

### Check VM Configuration:

**On Dom0:**
```bash
# Check Test-1's vGPU configuration
xe vm-param-get uuid=<test-1-uuid> param-name=platform param-key=device-model-args

# Should show: -device vgpu-stub,...,vm_id=1
```

**On VM:**
```bash
# Check what vm_id the VM sees
sudo ./vm_client_vector 0 0  # Will show the vm_id it reads from MMIO

# Output should show:
# [MMIO] Read vGPU properties:
#   VM ID: 1  ← Should match directory number
```

**Verify Directory Usage:**
```bash
# On Test-1, check which directory it uses
ls -la /mnt/vgpu/vm1/  # Should exist and be accessible

# On Dom0, check if requests arrive in correct directory
ls -la /var/vgpu/vm1/request.txt  # Should be created when Test-1 sends request
```

---

## SUMMARY

**The mapping is:**
- ✅ **Automatic** once configured (code uses `vm_id` to determine directory)
- ❌ **NOT automatic** in setup (you must configure each VM correctly)

**You must:**
1. Create directories manually (vm1/, vm2/, etc.)
2. Configure each VM with correct `vm_id` in vGPU stub
3. Ensure `vm_id` matches directory number (Test-1 → vm_id=1 → vm1/)

**The code then:**
- Reads `vm_id` from vGPU MMIO
- Uses it to construct directory paths automatically
- Ensures requests/responses go to correct directories

**Think of it as:**
- **Directories** = Physical locations (manually created)
- **vm_id** = Address/label (manually configured)
- **Code** = Delivery system (automatically routes based on vm_id)

---

**The mapping is enforced by code logic, but requires proper manual configuration!**
