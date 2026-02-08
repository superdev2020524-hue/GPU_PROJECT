# Connect XCP-ng Center via SSH Tunnel (Windows → Ubuntu → XCP-ng)

## Problem
- Windows PC cannot directly reach XCP-ng host (10.25.33.10)
- XCP-ng Center GUI requires network access to the host
- Solution: Use SSH port forwarding through Ubuntu machine

## Prerequisites
- Ubuntu machine can SSH to XCP-ng host (10.25.33.10) ✓ (you already do this)
- Windows PC can reach Ubuntu machine (need to verify)
- SSH client on Windows (PuTTY or built-in OpenSSH)

## Step-by-Step Instructions

### Step 1: Find Ubuntu Machine's IP Address

On Ubuntu machine, run:
```bash
ip addr show | grep "inet " | grep -v "127.0.0.1"
```

Note the IP address (e.g., `192.168.1.100` or similar).

### Step 2: Set Up SSH Tunnel (Choose One Method)

#### Method A: Using PuTTY on Windows (Recommended)

1. **Download PuTTY** (if not installed):
   - Download from:  
   - Install PuTTY

2. **Configure SSH Tunnel in PuTTY**:
   - Open PuTTY
   - **Host Name**: Enter your Ubuntu machine's IP address
   - **Port**: 22
   - **Connection Type**: SSH
   
3. **Set Up Port Forwarding**:
   - In left sidebar: **Connection → SSH → Tunnels**
   - **Source port**: `8443` (or any unused port)
   - **Destination**: `10.25.33.10:443`
   - Click **Add**
   - You should see: `L8443 10.25.33.10:443` in "Forwarded ports"
   
4. **Save Session** (optional):
   - Go back to **Session** in left sidebar
   - Enter name: `XCP-ng Tunnel`
   - Click **Save**
   
5. **Connect**:
   - Click **Open**
   - Login with your Ubuntu username/password
   - **Keep this window open** (tunnel stays active while PuTTY is connected)

#### Method B: Using Windows OpenSSH (Built-in)

1. **Open PowerShell or Command Prompt** on Windows

2. **Create SSH Tunnel**:
   ```cmd
   ssh -N -L 8443:10.25.33.10:443 <ubuntu-username>@<ubuntu-ip>
   ```
   
   Example:
   ```cmd
   ssh -N -L 8443:10.25.33.10:443 david@192.168.1.100
   ```
   
   - Enter Ubuntu password when prompted
   - **Keep this window open** (tunnel stays active)

### Step 3: Connect XCP-ng Center

1. **Open XCP-ng Center** on Windows

2. **Add New Server**:
   - Click **"Add a Server"** or **Server → Add**
   - **Server**: `localhost:8443` (or `127.0.0.1:8443`)
   - **User name**: `root`
   - **Password**: `Calvin@123` (from your info.txt)
   - Click **Add**

3. **Verify Connection**:
   - XCP-ng Center should connect successfully
   - You should see your host in the left pane

### Step 4: Create VGS ISO Storage Repository

Once connected in XCP-ng Center:

1. **Select your host** in the left pane (click the host name)

2. **Create New Storage Repository**:
   - Menu: **Storage → New Storage Repository...**
   - Or right-click host → **New Storage Repository...**

3. **Configure SR**:
   - **Type**: Select **"ISO library"** or **"File system (ISO library)"**
   - **Name**: `VGS ISO Storage`
   - **Location/Path**: `/mnt/iso-storage`
   - Click **Finish**

4. **Verify**:
   - The new SR should appear in the Storage list
   - It should show as mounted/accessible

### Step 5: Run VM Creation Script

Back on dom0 (via SSH from Ubuntu):

```bash
bash /home/david/Downloads/gpu/vm_create/create_test3_vm.sh
```

The script should now find the "VGS ISO Storage" SR and proceed successfully.

## Troubleshooting

### Tunnel Won't Connect
- Verify Ubuntu machine is reachable from Windows: `ping <ubuntu-ip>`
- Check SSH is running on Ubuntu: `sudo systemctl status ssh`
- Verify Ubuntu can reach XCP-ng: `ssh root@10.25.33.10` from Ubuntu

### XCP-ng Center Can't Connect Through Tunnel
- Verify tunnel is active (PuTTY/SSH window still open)
- Try `localhost:8443` instead of `127.0.0.1:8443`
- Check Windows firewall isn't blocking localhost connections

### SR Creation Fails in GUI
- Verify `/mnt/iso-storage` exists and is readable on dom0
- Check VGS volume is mounted: `mount | grep iso-storage`
- Try different SR type if "ISO library" doesn't work

## Alternative: Use Ubuntu Machine's IP Directly

If your Windows PC and Ubuntu machine are on the same network, you might be able to:

1. Set up SSH tunnel on Ubuntu that forwards to XCP-ng
2. Have Windows connect to Ubuntu's IP on the forwarded port

This requires running the tunnel script on Ubuntu:
```bash
bash /home/david/Downloads/gpu/vm_create/setup_xcp_center_tunnel.sh
```

Then in XCP-ng Center on Windows, connect to: `<ubuntu-ip>:8443`
