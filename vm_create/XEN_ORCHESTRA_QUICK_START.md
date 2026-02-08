# Xen Orchestra Quick Start (Web-Based XCP-ng Management)

## Why Xen Orchestra?
- **Works from Ubuntu browser** (no Windows required)
- **Full GUI** for creating ISO SRs, managing VMs, etc.
- **Web-based** - access from any machine that can reach it
- **Better than XCP-ng Center** in many ways

## Installation Options

### Option 1: XOA (Xen Orchestra Appliance) - EASIEST ⭐

**XOA is a pre-built VM** that you import into XCP-ng. It's the simplest method.

#### Steps:

1. **Download XOA**:
   - Visit: https://xen-orchestra.com/
   - Download the XOA OVA file (or use their installer)
   - File is usually ~500MB

2. **Import XOA as VM into XCP-ng** (via xe CLI from Ubuntu):
   ```bash
   # On Ubuntu, SSH to dom0 first
   ssh root@10.25.33.10
   
   # Then import XOA (you'll need the OVA file path)
   # This requires xe-import or similar tool
   ```

3. **Start XOA VM** and access via browser:
   - Default URL: `http://<xoa-vm-ip>:80`
   - Or `https://<xoa-vm-ip>:443`

4. **Connect XO to your XCP-ng host**:
   - In XO web interface, add server: `10.25.33.10`
   - Username: `root`
   - Password: `Calvin@123`

5. **Create VGS ISO Storage SR**:
   - In XO web interface: Storage → New → ISO library
   - Type: File system
   - Path: `/mnt/iso-storage`
   - Name: `VGS ISO Storage`

### Option 2: XO-from-source (More Control, More Complex)

Install Xen Orchestra directly on Ubuntu machine:

```bash
# On Ubuntu
git clone https://github.com/vatesfr/xen-orchestra
cd xen-orchestra
# Follow installation guide
npm install
npm run build
npm start
```

Then access: `http://localhost:8080`

**Pros:** Full control, runs on your Ubuntu machine
**Cons:** More setup, requires Node.js, npm, etc.

## Recommendation

**Use XOA (Option 1)** - it's pre-built and easier. Once XOA VM is running in XCP-ng, you can:
- Access it from Ubuntu browser
- Create the VGS ISO Storage SR via web GUI
- Then run `create_test3_vm.sh` and it will work

## Alternative: If You Want to Skip GUI Entirely

If installing XO is too much work right now, we could:
1. Try to manually create ISO SR using lower-level XAPI calls (advanced, risky)
2. Wait until SMB server is fixed
3. Use a different ISO source (if you have one)

But XOA is really the cleanest solution for your use case.
