# VM Directory Mapping Reference
**Date:** 2026-02-08  
**Purpose:** Explicit mapping between NFS directories and actual VMs

---

## DIRECTORY TO VM MAPPING

| NFS Directory | VM Name | Pool Assignment | Priority | VM ID | IP Address |
|--------------|---------|-----------------|----------|-------|------------|
| `/var/vgpu/vm1/` | **Test-1** | Pool A | High | 1 | 10.25.33.11 |
| `/var/vgpu/vm2/` | **Test-2** | Pool A | High | 2 | 10.25.33.12 |
| `/var/vgpu/vm3/` | **Test-3** | Pool A | Medium | 3 | 10.25.33.13 |
| `/var/vgpu/vm4/` | **Test-4** | Pool B | High | 4 | 10.25.33.14 |
| `/var/vgpu/vm5/` | **Test-5** | Pool B | Medium | 5 | 10.25.33.15 |
| `/var/vgpu/vm6/` | **Test-6** | Pool B | Low | 6 | 10.25.33.16 |
| `/var/vgpu/vm7/` | **Test-7** | Any Pool | Any | 7 | 10.25.33.17 |

---

## POOL ASSIGNMENTS

### Pool A (VMs 1-3)
- **Test-1** → `vm1/` directory
- **Test-2** → `vm2/` directory  
- **Test-3** → `vm3/` directory

### Pool B (VMs 4-6)
- **Test-4** → `vm4/` directory
- **Test-5** → `vm5/` directory
- **Test-6** → `vm6/` directory

### VM-7 (Configurable)
- **Test-7** → `vm7/` directory
- Can be assigned to Pool A or Pool B
- Priority can be set to any level

---

## USAGE EXAMPLES

### On Test-1 VM:
```bash
# Mount NFS
sudo mount -t nfs 10.25.33.10:/var/vgpu /mnt/vgpu

# Use vm1 directory
ls /mnt/vgpu/vm1/
echo "request" > /mnt/vgpu/vm1/request.txt
cat /mnt/vgpu/vm1/response.txt
```

### On Test-4 VM:
```bash
# Mount NFS
sudo mount -t nfs 10.25.33.10:/var/vgpu /mnt/vgpu

# Use vm4 directory
ls /mnt/vgpu/vm4/
echo "request" > /mnt/vgpu/vm4/request.txt
cat /mnt/vgpu/vm4/response.txt
```

### On Dom0 (MEDIATOR):
```bash
# MEDIATOR reads from all directories
ls /var/vgpu/vm*/request.txt

# MEDIATOR writes responses
echo "result" > /var/vgpu/vm1/response.txt
echo "result" > /var/vgpu/vm4/response.txt
```

---

## VERIFICATION

### Check Directory Mapping on Dom0:
```bash
# List all VM directories
ls -la /var/vgpu/

# Check README in each directory (if created)
for i in {1..7}; do
    echo "=== vm$i/ ==="
    cat /var/vgpu/vm$i/README.txt 2>/dev/null || echo "No README"
done
```

### Verify from VM:
```bash
# On Test-1, check vm1 directory exists
ls /mnt/vgpu/vm1/

# On Test-4, check vm4 directory exists
ls /mnt/vgpu/vm4/
```

---

## IMPORTANT NOTES

1. **VM ID Must Match Directory Number**
   - Test-1 must use `vm1/` and have `vm_id=1` in vGPU config
   - Test-4 must use `vm4/` and have `vm_id=4` in vGPU config
   - Mismatch will cause communication failures

2. **Each VM Uses Only Its Directory**
   - Test-1 should NOT write to `vm2/` or `vm4/`
   - Each VM has its own isolated directory

3. **MEDIATOR Reads All Directories**
   - MEDIATOR polls all `vm*/request.txt` files
   - It identifies the VM from the request content: `pool_id:priority:vm_id:num1:num2`
   - The directory name (`vm1`, `vm2`, etc.) should match the `vm_id` in the request

4. **File Structure**
   - `request.txt` - VM writes requests here
   - `response.txt` - MEDIATOR writes responses here
   - Both files are cleared after processing

---

## TROUBLESHOOTING

### "Directory not found" error:
- Check VM is using correct directory number
- Test-1 → `vm1/`, not `vm2/` or `vm4/`

### "Permission denied" error:
- Check directory permissions: `ls -la /var/vgpu/vm*/`
- Should be `drwxrwxrwx` (777)

### "Wrong VM ID" in requests:
- Verify vGPU config matches directory number
- Test-1 should have `vm_id=1` in vGPU stub device

---

**This mapping ensures clear communication between VMs and MEDIATOR!**
