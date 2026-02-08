# Install Xen Orchestra Appliance (XOA) - Step by Step

## Goal
Install XOA as a VM in XCP-ng so you can manage XCP-ng from Ubuntu browser and create the VGS ISO Storage SR.

## Prerequisites
- Ubuntu machine with SSH access to XCP-ng dom0 (10.25.33.10) ✓
- At least 2GB RAM and 20GB disk space available on XCP-ng for XOA VM
- Web browser on Ubuntu

## Step 1: Download XOA

### Option A: Download XOA OVA file (if available)
Visit: https://xen-orchestra.com/#!/downloads
- Look for "XOA" or "Xen Orchestra Appliance"
- Download the OVA or XVA file

### Option B: Use XOA installer script (easier)
XOA provides an installation script that can be run from dom0.

## Step 2: Install XOA on XCP-ng (Recommended Method)

### ⚠️ UPDATE: Official installer didn't work
The official installer script (`bash <(curl -s https://xoa.io/install.sh)`) produced no output.
The direct XVA download also returned 404.

### ✅ ALTERNATIVE: Use Docker Method (Easier)
See: `XOA_DOCKER_METHOD.md` for the simplest installation method.

Docker method runs XO on Ubuntu (not as VM in XCP-ng), but it can still manage XCP-ng over the network.

### Original Method (If you want to try manual installation):
If you still want XOA as a VM in XCP-ng, see Step 3 below for manual method.

## Step 3: Manual XOA Installation (If Script Fails)

If the installer script doesn't work, we can create XOA VM manually:

### 3.1: Download XOA XVA file
```bash
# On Ubuntu
wget https://xoa.io/releases/xoa-vm.xva
# Or find the latest download link from xen-orchestra.com
```

### 3.2: Transfer to dom0 and import
```bash
# From Ubuntu, copy to dom0
scp xoa-vm.xva root@10.25.33.10:/root/

# SSH to dom0
ssh root@10.25.33.10

# Import XVA as VM
xe vm-import filename=/root/xoa-vm.xva
```

### 3.3: Configure XOA VM
```bash
# Find the imported VM UUID
XOA_VM_UUID=$(xe vm-list name-label="XOA" params=uuid --minimal | head -1)

# Set memory (XOA needs at least 2GB)
xe vm-memory-limits-set uuid=$XOA_VM_UUID static-min=2GiB static-max=2GiB dynamic-min=2GiB dynamic-max=2GiB

# Set VCPUs
xe vm-param-set uuid=$XOA_VM_UUID VCPUs-max=2 VCPUs-at-startup=2

# Attach to network
NET_UUID=$(xe network-list bridge=xenbr0 params=uuid --minimal | head -1)
VIF_UUID=$(xe vif-create vm-uuid=$XOA_VM_UUID network-uuid=$NET_UUID device=0)

# Start VM
xe vm-start uuid=$XOA_VM_UUID
```

### 3.4: Get XOA VM IP address
```bash
# Wait a moment for VM to boot
sleep 30

# Get VM's IP (XOA usually gets IP via DHCP)
xe vm-list uuid=$XOA_VM_UUID params=name-label,power-state
# Check XOA console or use VNC to see IP, or check DHCP leases
```

## Step 4: Access Xen Orchestra Web Interface

### 4.1: Find XOA VM IP
XOA will get an IP via DHCP on the same network (10.25.33.x).

**Method 1: Check from dom0**
```bash
# List running VMs and their network info
xe vm-list is-control-domain=false params=name-label,power-state
# Then check VIF MAC and match to DHCP leases, or use VNC to see XOA console
```

**Method 2: Use VNC to see XOA console**
```bash
# Get XOA VM dom-id
XOA_DOMID=$(xe vm-param-get uuid=$XOA_VM_UUID param-name=dom-id)

# Connect via VNC to see XOA's console (will show IP)
# Use same VNC method as for other VMs
```

**Method 3: Scan network (from Ubuntu)**
```bash
# From Ubuntu, scan for XOA (if you know it's on 10.25.33.x)
nmap -p 80,443 10.25.33.0/24
# Look for open ports 80/443 that aren't dom0
```

### 4.2: Access XO Web Interface
Once you have XOA IP (let's say it's `10.25.33.50`):

1. **Open browser on Ubuntu**
2. **Navigate to**: `http://10.25.33.50` or `https://10.25.33.50`
3. **Default credentials** (if prompted):
   - Username: `admin@admin.net`
   - Password: `admin`

## Step 5: Connect XO to Your XCP-ng Host

1. **In XO web interface**, click **"Add Server"** or **"New" → "Server"**
2. **Enter connection details**:
   - **Host**: `10.25.33.10`
   - **Username**: `root`
   - **Password**: `Calvin@123`
3. **Click "Add"** or "Connect"
4. You should see your XCP-ng host appear in XO

## Step 6: Create VGS ISO Storage SR via XO

1. **In XO web interface**, select your host (`10.25.33.10`)
2. **Go to Storage tab** or **Storage → New**
3. **Create New Storage Repository**:
   - **Type**: Select **"ISO library"** or **"File system (ISO library)"**
   - **Name**: `VGS ISO Storage`
   - **Location/Path**: `/mnt/iso-storage`
   - Click **"Create"** or **"Finish"**
4. **Verify SR is created and mounted**:
   - Should appear in Storage list
   - Should show as "attached" or "mounted"

## Step 7: Verify and Test

### On dom0, verify SR exists:
```bash
xe sr-list name-label="VGS ISO Storage" params=uuid,name-label,type,content-type
```

### Run VM creation script:
```bash
bash /home/david/Downloads/gpu/vm_create/create_test3_vm.sh
```

The script should now find "VGS ISO Storage" SR and proceed successfully!

## Troubleshooting

### XOA VM won't start
- Check: `xe vm-list uuid=$XOA_VM_UUID params=power-state`
- Check logs: `tail -50 /var/log/xensource.log | grep -i xoa`
- Verify memory/disk available: `xe host-compute-free`

### Can't access XO web interface
- Verify XOA VM is running: `xe vm-list name-label="XOA"`
- Check XOA got IP (use VNC to see console)
- Try both HTTP (port 80) and HTTPS (port 443)
- Check firewall on dom0 isn't blocking

### XO can't connect to XCP-ng host
- Verify XOA VM can reach 10.25.33.10 (ping from XOA console)
- Check XAPI is running: `systemctl status xapi`
- Verify XAPI port 443 is open: `ss -tlnp | grep 443`

## Next Steps After XOA is Working

Once XOA is installed and you've created the VGS ISO Storage SR:
1. Run `create_test3_vm.sh` - it will find the new SR
2. VM creation should proceed successfully
3. Continue with VNC setup and Ubuntu installation
