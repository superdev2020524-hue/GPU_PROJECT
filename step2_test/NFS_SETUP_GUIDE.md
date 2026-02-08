# NFS Setup Guide - Complete Instructions
**Date:** 2026-02-08  
**Purpose:** Enable NFS communication between Dom0 (host) and VMs (guests)

---

## OVERVIEW

NFS (Network File System) allows VMs to access a shared directory on Dom0 for communication:
- **Dom0 (Host):** Exports `/var/vgpu` directory via NFS
- **VM (Guest):** Mounts the exported directory at `/mnt/vgpu`

This enables file-based communication:
- VM writes requests to `/mnt/vgpu/vm<id>/request.txt`
- MEDIATOR reads from `/var/vgpu/vm<id>/request.txt`
- MEDIATOR writes responses to `/var/vgpu/vm<id>/response.txt`
- VM reads from `/mnt/vgpu/vm<id>/response.txt`

---

## PART 1: DOM0 (HOST) SETUP

### Step 1.1: Install NFS Server Packages

**On Dom0 (XCP-ng is based on CentOS):**

```bash
# Check if NFS is already installed
rpm -qa | grep nfs-utils

# If not installed, install it
yum install -y nfs-utils rpcbind

# Enable and start services
systemctl enable rpcbind
systemctl enable nfs-server
systemctl start rpcbind
systemctl start nfs-server

# Verify services are running
systemctl status rpcbind
systemctl status nfs-server
```

**Expected Output:**
```
● rpcbind.service - RPC bind service
   Loaded: loaded (...)
   Active: active (running) since ...
```

---

### Step 1.2: Create Shared Directory Structure

**IMPORTANT: VM Directory Mapping**

The NFS directories map to actual VMs as follows:
- `vm1/` → **Test-1** (Pool A, VM ID: 1)
- `vm2/` → **Test-2** (Pool A, VM ID: 2)
- `vm3/` → **Test-3** (Pool A, VM ID: 3)
- `vm4/` → **Test-4** (Pool B, VM ID: 4)
- `vm5/` → **Test-5** (Pool B, VM ID: 5)
- `vm6/` → **Test-6** (Pool B, VM ID: 6)
- `vm7/` → **Test-7** (Any Pool, VM ID: 7)

**On Dom0:**

```bash
# Create base directory
sudo mkdir -p /var/vgpu
sudo chmod 777 /var/vgpu

# Create per-VM directories (for VMs 1-7)
# Each directory corresponds to a specific VM
for i in {1..7}; do
    sudo mkdir -p /var/vgpu/vm$i
    sudo chmod 777 /var/vgpu/vm$i
    
    # Initialize response file (so VMs can read it)
    echo "0:Ready" | sudo tee /var/vgpu/vm$i/response.txt > /dev/null
    sudo chmod 666 /var/vgpu/vm$i/response.txt
    
    # Create a README to document which VM this is
    case $i in
        1) VM_NAME="Test-1 (Pool A)";;
        2) VM_NAME="Test-2 (Pool A)";;
        3) VM_NAME="Test-3 (Pool A)";;
        4) VM_NAME="Test-4 (Pool B)";;
        5) VM_NAME="Test-5 (Pool B)";;
        6) VM_NAME="Test-6 (Pool B)";;
        7) VM_NAME="Test-7 (Any Pool)";;
    esac
    echo "# NFS Directory for $VM_NAME" | sudo tee /var/vgpu/vm$i/README.txt > /dev/null
    echo "# VM ID: $i" | sudo tee -a /var/vgpu/vm$i/README.txt > /dev/null
done

# Verify structure
ls -la /var/vgpu/
ls -la /var/vgpu/vm*/
```

**Expected Structure:**
```
/var/vgpu/
├── vm1/          → Test-1 (Pool A, VM ID: 1)
│   ├── README.txt
│   └── response.txt
├── vm2/          → Test-2 (Pool A, VM ID: 2)
│   ├── README.txt
│   └── response.txt
├── vm3/          → Test-3 (Pool A, VM ID: 3)
│   ├── README.txt
│   └── response.txt
├── vm4/          → Test-4 (Pool B, VM ID: 4)
│   ├── README.txt
│   └── response.txt
├── vm5/          → Test-5 (Pool B, VM ID: 5)
│   ├── README.txt
│   └── response.txt
├── vm6/          → Test-6 (Pool B, VM ID: 6)
│   ├── README.txt
│   └── response.txt
└── vm7/          → Test-7 (Any Pool, VM ID: 7)
    ├── README.txt
    └── response.txt
```

---

### Step 1.3: Configure NFS Export

**On Dom0:**

```bash
# Edit exports file
sudo nano /etc/exports
# or
sudo vi /etc/exports
```

**Add this line to `/etc/exports`:**
```
/var/vgpu *(rw,sync,no_root_squash,no_subtree_check,fsid=1,insecure)
```

**Explanation of options:**
- `/var/vgpu` - Directory to export
- `*` - Allow access from any IP (restrict to specific IPs for security: `10.25.33.0/24`)
- `rw` - Read-write access
- `sync` - Synchronous writes (more reliable)
- `no_root_squash` - Allow root access (needed for VM operations)
- `no_subtree_check` - Faster, less secure but OK for this use case
- `fsid=1` - Filesystem ID
- `insecure` - Allow connections from ports > 1024

**For better security (restrict to VM network):**
```
/var/vgpu 10.25.33.0/24(rw,sync,no_root_squash,no_subtree_check,fsid=1,insecure)
```

---

### Step 1.4: Apply NFS Export Configuration

**On Dom0:**

```bash
# Export the directory
sudo exportfs -rav

# Verify export
sudo exportfs -v
```

**Expected Output:**
```
/var/vgpu        <world>(sync,wdelay,hide,no_subtree_check,sec=sys,rw,secure,no_root_squash,no_all_squash)
```

**Check NFS is listening:**
```bash
# Check RPC services
rpcinfo -p localhost

# Should show:
#   100000    2   tcp    111  portmapper
#   100000    2   udp    111  portmapper
#   100003    3   tcp   2049  nfs
#   100003    4   tcp   2049  nfs
#   100227    3   tcp   2049  nfs_acl
```

---

### Step 1.5: Configure Firewall (if enabled)

**On Dom0:**

```bash
# Check if firewall is running
systemctl status firewalld

# If firewall is active, allow NFS
firewall-cmd --permanent --add-service=nfs
firewall-cmd --permanent --add-service=rpc-bind
firewall-cmd --permanent --add-service=mountd
firewall-cmd --reload

# Or disable firewall temporarily for testing
systemctl stop firewalld
systemctl disable firewalld
```

---

### Step 1.6: Verify Dom0 Setup

**On Dom0:**

```bash
# Test local mount (optional)
sudo mount -t nfs localhost:/var/vgpu /mnt/test
ls /mnt/test
sudo umount /mnt/test

# Check directory permissions
ls -la /var/vgpu/
```

**Everything should be ready on Dom0!**

---

## PART 2: VM (GUEST) SETUP

### Step 2.1: Install NFS Client Packages

**On VM (Ubuntu/Debian):**

```bash
# Update package list
sudo apt-get update

# Install NFS client
sudo apt-get install -y nfs-common

# Verify installation
which mount.nfs
```

**On VM (RHEL/CentOS):**

```bash
# Install NFS client
sudo yum install -y nfs-utils

# Verify installation
which mount.nfs
```

---

### Step 2.2: Find Dom0 IP Address

**On VM:**

```bash
# Option 1: If you know the IP
DOM0_IP="10.25.33.10"  # Replace with your Dom0 IP

# Option 2: Find gateway (usually Dom0)
ip route | grep default
# Example output: default via 10.25.33.10 dev eth0

# Option 3: Ping common Dom0 IPs
ping -c 1 10.25.33.10
```

**Note:** Dom0 IP is typically the gateway IP for VMs.

---

### Step 2.3: Test NFS Connection

**On VM:**

```bash
# Replace with your Dom0 IP
DOM0_IP="10.25.33.10"

# Test if NFS is accessible
showmount -e $DOM0_IP
```

**Expected Output:**
```
Export list for 10.25.33.10:
/var/vgpu *
```

**If this fails:**
- Check Dom0 firewall
- Verify NFS services are running on Dom0
- Check network connectivity: `ping $DOM0_IP`

---

### Step 2.4: Create Mount Point

**On VM:**

```bash
# Create mount point
sudo mkdir -p /mnt/vgpu

# Verify
ls -ld /mnt/vgpu
```

---

### Step 2.5: Mount NFS Share

**On VM (Manual Mount - for testing):**

```bash
# Replace with your Dom0 IP
DOM0_IP="10.25.33.10"

# Mount NFS share
sudo mount -t nfs $DOM0_IP:/var/vgpu /mnt/vgpu

# Verify mount
mount | grep vgpu
ls -la /mnt/vgpu/
```

**Expected Output:**
```
10.25.33.10:/var/vgpu on /mnt/vgpu type nfs (rw,relatime,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,hard,proto=tcp,timeo=600,retrans=2,sec=sys,clientaddr=10.25.33.13,local_lock=none,addr=10.25.33.10)

/mnt/vgpu/
total 0
drwxrwxrwx 1 root root  0 Feb  8 08:30 .
drwxr-xr-x 1 root root 20 Feb  8 08:30 ..
drwxrwxrwx 1 root root  0 Feb  8 08:30 vm1
drwxrwxrwx 1 root root  0 Feb  8 08:30 vm2
...
```

---

### Step 2.6: Configure Automatic Mount (Optional but Recommended)

**On VM (Ubuntu/Debian):**

```bash
# Edit fstab
sudo nano /etc/fstab
# or
sudo vi /etc/fstab
```

**Add this line (replace with your Dom0 IP):**
```
10.25.33.10:/var/vgpu  /mnt/vgpu  nfs  defaults,_netdev  0  0
```

**Test fstab entry:**
```bash
# Test mount (doesn't actually mount, just validates)
sudo mount -a

# If no errors, it's configured correctly
```

**On VM (RHEL/CentOS):**

Same as above - edit `/etc/fstab` with the same entry.

---

### Step 2.7: Verify VM Setup

**On VM:**

**IMPORTANT: Use the correct directory for your VM!**
- **Test-1** → Use `/mnt/vgpu/vm1/`
- **Test-2** → Use `/mnt/vgpu/vm2/`
- **Test-3** → Use `/mnt/vgpu/vm3/`
- **Test-4** → Use `/mnt/vgpu/vm4/`
- **Test-5** → Use `/mnt/vgpu/vm5/`
- **Test-6** → Use `/mnt/vgpu/vm6/`
- **Test-7** → Use `/mnt/vgpu/vm7/`

```bash
# Check mount
mount | grep vgpu

# Determine your VM number from VM name
# If you're on Test-1, VM_ID=1; Test-2, VM_ID=2; etc.
VM_ID=1  # Replace with your actual VM number (1-7)

# Test write access to YOUR VM's directory
echo "test" | sudo tee /mnt/vgpu/vm${VM_ID}/test.txt
cat /mnt/vgpu/vm${VM_ID}/test.txt
sudo rm /mnt/vgpu/vm${VM_ID}/test.txt

# Check your VM's directory exists
ls -la /mnt/vgpu/vm${VM_ID}/

# Read the README to confirm mapping
cat /mnt/vgpu/vm${VM_ID}/README.txt
```

**If write fails:**
- Check permissions on Dom0: `ls -la /var/vgpu/vm*/`
- Verify NFS export allows write: `cat /etc/exports`
- Check SELinux (if enabled): `getenforce`

---

## PART 3: TROUBLESHOOTING

### Problem: "mount.nfs: access denied by server"

**Solution:**
```bash
# On Dom0: Check exports
cat /etc/exports
sudo exportfs -rav

# Check firewall
systemctl status firewalld
firewall-cmd --list-all

# Check NFS services
systemctl status nfs-server
systemctl status rpcbind
```

---

### Problem: "mount.nfs: Connection refused"

**Solution:**
```bash
# On Dom0: Start NFS services
sudo systemctl start rpcbind
sudo systemctl start nfs-server

# Check if services are listening
rpcinfo -p localhost
netstat -tuln | grep 2049
```

---

### Problem: "Permission denied" when writing files

**Solution:**
```bash
# On Dom0: Fix permissions
sudo chmod 777 /var/vgpu
sudo chmod 777 /var/vgpu/vm*

# Check exports has 'no_root_squash'
grep no_root_squash /etc/exports
```

---

### Problem: NFS mount disconnects after reboot

**Solution:**
```bash
# On VM: Add to /etc/fstab (see Step 2.6)
# Use _netdev option to wait for network

# Or create systemd mount unit (advanced)
```

---

### Problem: "showmount: command not found"

**Solution:**
```bash
# On VM: Install nfs-utils
# Ubuntu/Debian:
sudo apt-get install nfs-common

# RHEL/CentOS:
sudo yum install nfs-utils
```

---

## PART 4: QUICK SETUP SCRIPT

### Dom0 Setup Script

**Save as `setup_nfs_dom0.sh` on Dom0:**

```bash
#!/bin/bash
# NFS Setup Script for Dom0

echo "Setting up NFS on Dom0..."

# Install packages
yum install -y nfs-utils rpcbind

# Create directories
mkdir -p /var/vgpu
chmod 777 /var/vgpu

# Create directories for each VM with proper mapping
# vm1 → Test-1 (Pool A), vm2 → Test-2 (Pool A), vm3 → Test-3 (Pool A)
# vm4 → Test-4 (Pool B), vm5 → Test-5 (Pool B), vm6 → Test-6 (Pool B)
# vm7 → Test-7 (Any Pool)
for i in {1..7}; do
    mkdir -p /var/vgpu/vm$i
    chmod 777 /var/vgpu/vm$i
    echo "0:Ready" > /var/vgpu/vm$i/response.txt
    chmod 666 /var/vgpu/vm$i/response.txt
    
    # Document which VM this directory is for
    case $i in
        1) echo "# Test-1 (Pool A, VM ID: 1)" > /var/vgpu/vm$i/README.txt;;
        2) echo "# Test-2 (Pool A, VM ID: 2)" > /var/vgpu/vm$i/README.txt;;
        3) echo "# Test-3 (Pool A, VM ID: 3)" > /var/vgpu/vm$i/README.txt;;
        4) echo "# Test-4 (Pool B, VM ID: 4)" > /var/vgpu/vm$i/README.txt;;
        5) echo "# Test-5 (Pool B, VM ID: 5)" > /var/vgpu/vm$i/README.txt;;
        6) echo "# Test-6 (Pool B, VM ID: 6)" > /var/vgpu/vm$i/README.txt;;
        7) echo "# Test-7 (Any Pool, VM ID: 7)" > /var/vgpu/vm$i/README.txt;;
    esac
done

# Configure export
if ! grep -q "/var/vgpu" /etc/exports; then
    echo "/var/vgpu *(rw,sync,no_root_squash,no_subtree_check,fsid=1,insecure)" >> /etc/exports
fi

# Start services
systemctl enable rpcbind nfs-server
systemctl start rpcbind nfs-server

# Export
exportfs -rav

# Firewall (if enabled)
if systemctl is-active --quiet firewalld; then
    firewall-cmd --permanent --add-service=nfs
    firewall-cmd --permanent --add-service=rpc-bind
    firewall-cmd --permanent --add-service=mountd
    firewall-cmd --reload
fi

echo "NFS setup complete!"
echo "Export: /var/vgpu"
exportfs -v
```

**Run:**
```bash
chmod +x setup_nfs_dom0.sh
sudo ./setup_nfs_dom0.sh
```

---

### VM Setup Script

**Save as `setup_nfs_vm.sh` on VM:**

```bash
#!/bin/bash
# NFS Setup Script for VM

DOM0_IP="${1:-10.25.33.10}"  # Default Dom0 IP, or pass as argument

echo "Setting up NFS client on VM..."
echo "Dom0 IP: $DOM0_IP"

# Install packages (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y nfs-common
# RHEL/CentOS
elif command -v yum &> /dev/null; then
    sudo yum install -y nfs-utils
fi

# Create mount point
sudo mkdir -p /mnt/vgpu

# Test connection
echo "Testing NFS connection..."
if showmount -e $DOM0_IP &> /dev/null; then
    echo "✓ NFS accessible"
else
    echo "✗ Cannot access NFS. Check Dom0 IP and firewall."
    exit 1
fi

# Mount
echo "Mounting NFS..."
sudo mount -t nfs $DOM0_IP:/var/vgpu /mnt/vgpu

# Verify
if mount | grep -q vgpu; then
    echo "✓ NFS mounted successfully"
    ls -la /mnt/vgpu/
else
    echo "✗ Mount failed"
    exit 1
fi

# Add to fstab (optional)
read -p "Add to /etc/fstab for automatic mount? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if ! grep -q "/mnt/vgpu" /etc/fstab; then
        echo "$DOM0_IP:/var/vgpu  /mnt/vgpu  nfs  defaults,_netdev  0  0" | sudo tee -a /etc/fstab
        echo "✓ Added to /etc/fstab"
    else
        echo "Already in /etc/fstab"
    fi
fi

echo "NFS client setup complete!"
```

**Run:**
```bash
chmod +x setup_nfs_vm.sh
./setup_nfs_vm.sh [DOM0_IP]
# Example: ./setup_nfs_vm.sh 10.25.33.10
```

---

## PART 5: VERIFICATION CHECKLIST

### Dom0 Checklist:
- [ ] NFS packages installed (`nfs-utils`, `rpcbind`)
- [ ] Services running (`rpcbind`, `nfs-server`)
- [ ] `/var/vgpu` directory created with proper permissions
- [ ] Per-VM directories created (vm1-vm7)
- [ ] `/etc/exports` configured
- [ ] Export active (`exportfs -v` shows `/var/vgpu`)
- [ ] Firewall configured (if enabled)
- [ ] Can see exports: `showmount -e localhost`

### VM Checklist:
- [ ] NFS client packages installed (`nfs-common` or `nfs-utils`)
- [ ] Dom0 IP identified
- [ ] Can see exports: `showmount -e <DOM0_IP>`
- [ ] Mount point created (`/mnt/vgpu`)
- [ ] NFS mounted successfully
- [ ] Can read from mount: `ls /mnt/vgpu/`
- [ ] Can write to mount: `echo test > /mnt/vgpu/vm1/test.txt`
- [ ] (Optional) Added to `/etc/fstab` for auto-mount

---

## SUMMARY

**Dom0 (Host):**
1. Install `nfs-utils` and `rpcbind`
2. Create `/var/vgpu` and per-VM directories (vm1-vm7)
   - **Important:** Each directory maps to a specific VM (see VM_DIRECTORY_MAPPING.md)
3. Add to `/etc/exports`: `/var/vgpu *(rw,sync,no_root_squash,...)`
4. Start services: `systemctl start rpcbind nfs-server`
5. Export: `exportfs -rav`

**VM (Guest):**
1. Install `nfs-common` (Ubuntu) or `nfs-utils` (RHEL)
2. Find Dom0 IP (usually gateway)
3. Test: `showmount -e <DOM0_IP>`
4. Mount: `mount -t nfs <DOM0_IP>:/var/vgpu /mnt/vgpu`
5. **Use correct directory:** Test-1 → `vm1/`, Test-4 → `vm4/`, etc.
6. (Optional) Add to `/etc/fstab` for auto-mount

**VM Directory Mapping:**
- See `VM_DIRECTORY_MAPPING.md` for complete mapping
- Each VM must use its corresponding directory (Test-1 → vm1/, Test-4 → vm4/, etc.)

**That's it! NFS is now ready for MEDIATOR-VM communication.**
