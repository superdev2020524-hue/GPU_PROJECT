# Install Xen Orchestra via Docker (Simplest Method)

## Why Docker?
- Official XOA installer didn't work
- XVA download returned 404
- Docker method is simpler and works on Ubuntu
- XO running in Docker can still manage XCP-ng over network

## Step-by-Step Installation

### Step 1: Install Docker on Ubuntu

```bash
# Update package list
sudo apt update

# Install Docker
sudo apt install -y docker.io

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Verify Docker is working
sudo docker --version
```

### Step 2: Run Xen Orchestra Container

```bash
# Run XOA container (community edition)
sudo docker run -d \
  --name xoa \
  -p 80:80 \
  -p 443:443 \
  --restart=unless-stopped \
  ronivay/xen-orchestra

# Check if container is running
sudo docker ps | grep xoa
```

### Step 3: Access Xen Orchestra

1. **Open browser on Ubuntu**
2. **Navigate to**: `http://localhost` or `http://127.0.0.1`
3. **Default credentials** (if prompted):
   - Username: `admin@admin.net`
   - Password: `admin`

### Step 4: Connect XO to Your XCP-ng Host

**📖 See detailed guide**: `CONNECT_XCP_HOST_IN_XO.md` for step-by-step instructions with troubleshooting.

**Quick steps:**
1. In XO web interface, click **"+ Connect pool"** button (top right corner)
2. Enter:
   - **Host**: `10.25.33.10`
   - **Username**: `root`
   - **Password**: `Calvin@123`
   - **Port**: `443` (default)
3. Click **"Connect"** or **"Add"**
4. Wait for connection (10-30 seconds)
5. Verify: Dashboard should show "Connected 1" in Pools status

### Step 5: Create VGS ISO Storage SR

1. Select your host (`10.25.33.10`) in XO
2. Go to **Storage** tab
3. Click **"New"** or **"Add Storage"**
4. Select **"ISO library"** or **"File system (ISO library)"**
5. Configure:
   - **Name**: `VGS ISO Storage`
   - **Path**: `/mnt/iso-storage`
6. Click **"Create"**

### Step 6: Verify and Test

```bash
# On dom0, verify SR was created
ssh root@10.25.33.10
xe sr-list name-label="VGS ISO Storage" params=uuid,name-label,type
```

Then run:
```bash
bash /home/david/Downloads/gpu/vm_create/create_test3_vm.sh
```

## Managing XOA Container

### Stop XOA

```bash
# Stop the container (XO will be unavailable)
sudo docker stop xoa

# Verify it's stopped
sudo docker ps -a | grep xoa
```

### Start XOA

```bash
# Start the container (if it was stopped)
sudo docker start xoa

# Verify it's running
sudo docker ps | grep xoa

# Check container logs
sudo docker logs xoa
```

### Restart XOA

```bash
# Restart the container (useful after configuration changes)
sudo docker restart xoa

# Verify it's running
sudo docker ps | grep xoa
```

### Remove XOA Container

```bash
# Stop the container first
sudo docker stop xoa

# Remove the container (this does NOT delete your XO data)
sudo docker rm xoa

# To remove with data volumes (WARNING: This deletes all XO data!)
sudo docker rm -v xoa
```

### Update XOA to Latest Version

```bash
# Stop the container
sudo docker stop xoa

# Remove the old container
sudo docker rm xoa

# Pull the latest image
sudo docker pull ronivay/xen-orchestra:latest

# Run the new container (with same settings as before)
sudo docker run -d \
  --name xoa \
  -p 80:80 \
  -p 443:443 \
  --restart=unless-stopped \
  ronivay/xen-orchestra:latest
```

### Check XOA Status

```bash
# Check if container is running
sudo docker ps | grep xoa

# Check container logs (last 50 lines)
sudo docker logs --tail 50 xoa

# Check container resource usage
sudo docker stats xoa
```

### Persistent Data Storage

The XOA container stores its data in Docker volumes. To use persistent storage (recommended):

```bash
# Stop and remove existing container
sudo docker stop xoa
sudo docker rm xoa

# Run with persistent volumes
sudo docker run -d \
  --name xoa \
  -p 80:80 \
  -p 443:443 \
  --restart=unless-stopped \
  -v /var/lib/xo-server:/var/lib/xo-server \
  -v /var/lib/xo-server/xo-data:/var/lib/xo-server/xo-data \
  ronivay/xen-orchestra:latest
```

This ensures your XO configuration and data persist even if you remove and recreate the container.

## Troubleshooting

### Container won't start
```bash
# Check Docker logs
sudo docker logs xoa

# Check if ports are in use
sudo netstat -tlnp | grep -E ":(80|443)"
```

### Can't access web interface
- Try: `http://localhost:80` or `https://localhost:443`
- Check firewall: `sudo ufw status`
- Check container is running: `sudo docker ps`

### XO can't connect to XCP-ng
- Verify from Ubuntu: `ping 10.25.33.10`
- Check XAPI is running on dom0: `ssh root@10.25.33.10 'systemctl status xapi'`
