# Connect XCP-ng Host to Xen Orchestra - Detailed Guide

## Current Status
✅ You have successfully:
- Installed Xen Orchestra via Docker
- Accessed the web interface at `http://localhost`
- Completed the registration/login process
- Can see the dashboard (showing 0 pools, 0 hosts, 0 VMs)

## Next Step: Connect Your XCP-ng Host

### Step 1: Click the "+ Connect pool" Button

**Location**: Look at the **top right corner** of the dashboard page, next to the tabs (DASHBOARD, BACKUPS, TASKS, POOLS, HOSTS, VMS).

You should see a button labeled **"+ Connect pool"** (it may be purple/blue colored).

**Action**: Click this button.

---

### Step 2: Fill in the Connection Form

After clicking "+ Connect pool", a dialog/modal window should appear asking for connection details.

**Fill in the following fields:**

1. **Host / Address / Server**:
   - Enter: `10.25.33.10`
   - (This is your XCP-ng dom0 IP address)

2. **Username**:
   - Enter: `root`
   - (This is the dom0 root username)

3. **Password**:
   - Enter: `Calvin@123`
   - (This is the dom0 root password)

4. **Port** (if shown):
   - Usually: `443` (default XAPI port)
   - If not shown, leave default or enter `443`

5. **Name / Label** (optional, if shown):
   - You can enter: `XCP-ng Host` or `10.25.33.10`
   - This is just a friendly name for display

6. **Other fields** (if any):
   - Leave defaults unless you know specific settings needed

---

### Step 3: Click "Connect" or "Add" Button

After filling in the form, look for a button at the bottom of the dialog:
- **"Connect"**
- **"Add"**
- **"Save"**
- **"OK"**

Click this button to establish the connection.

---

### Step 4: Wait for Connection

Xen Orchestra will attempt to connect to your XCP-ng host. This may take 10-30 seconds.

**What to expect:**
- A loading spinner or progress indicator
- The dialog may close automatically on success
- You may see a success message

**If successful:**
- The dashboard should update
- You should see:
  - **Pools status**: "Connected 1" (instead of 0)
  - **Hosts status**: "Running 1" (instead of 0)
  - The left sidebar may show your host/pool

---

### Step 5: Verify Connection

**Check the dashboard:**
- Look at the **"Pools status"** widget (top left)
- Should show: **"Connected 1"** (green dot)
- Look at the **"Hosts status"** widget (top middle)
- Should show: **"Running 1"** (green dot)

**Check the left sidebar:**
- You may see a new entry under the search bar
- It might show your host name or IP address

**Check the POOLS tab:**
- Click on the **"POOLS"** tab at the top
- You should see your XCP-ng pool listed

**Check the HOSTS tab:**
- Click on the **"HOSTS"** tab at the top
- You should see your host (`10.25.33.10`) listed

---

## Troubleshooting

### Problem: "+ Connect pool" button doesn't do anything

**Solution:**
- Try refreshing the page (`F5` or `Ctrl+R`)
- Check browser console for errors (`F12` → Console tab)
- Try a different browser

---

### Problem: Connection fails with "Connection refused" or "Cannot connect"

**Check 1: XAPI service on dom0**
```bash
# From Ubuntu, SSH to dom0 and check XAPI
ssh root@10.25.33.10
systemctl status xapi
```

**If XAPI is not running:**
```bash
systemctl start xapi
systemctl enable xapi
```

**Check 2: XAPI port 443 is listening**
```bash
# On dom0
ss -tlnp | grep 443
```

Should show something like:
```
LISTEN 0 128 *:443 *:* users:(("xapi",pid=1234,fd=123))
```

**Check 3: Network connectivity from Ubuntu**
```bash
# From Ubuntu (not dom0)
ping 10.25.33.10
telnet 10.25.33.10 443
```

**Check 4: Firewall on dom0**
```bash
# On dom0
iptables -L -n | grep 443
# If nothing shows, firewall might be blocking
```

---

### Problem: Connection fails with "Authentication failed" or "Invalid credentials"

**Solution:**
- Double-check username: `root` (lowercase)
- Double-check password: `Calvin@123` (case-sensitive, includes @ symbol)
- Try connecting from dom0 directly to verify credentials:
  ```bash
  # On dom0
  xe host-list
  ```
  If this works, credentials are correct.

---

### Problem: Connection succeeds but shows "0" hosts/VMs

**This is normal if:**
- Your XCP-ng host has no VMs yet
- The connection is still initializing (wait 30 seconds and refresh)

**To verify connection is working:**
- Click on **"HOSTS"** tab
- You should see your host listed
- Click on the host to see details

---

### Problem: Dialog/form doesn't appear after clicking "+ Connect pool"

**Solution:**
- Check if a popup blocker is enabled (disable it temporarily)
- Try right-clicking the button and selecting "Inspect Element" to see if there are JavaScript errors
- Check browser console (`F12` → Console) for errors
- Try a different browser (Firefox, Chrome, etc.)

---

## Alternative: Manual Connection via Settings

If the "+ Connect pool" button doesn't work, try this alternative:

1. **Look for a Settings/Configuration menu:**
   - Click on your **user profile icon** (top right, purple circle with "XO")
   - Or look for a **gear icon** or **"Settings"** menu

2. **Navigate to "Servers" or "Pools":**
   - Look for a section called "Servers", "Pools", "Connections", or "Hosts"

3. **Add new server:**
   - Click "Add Server" or "+" button
   - Fill in the same form as described above

---

## Next Steps After Successful Connection

Once you see your host connected (Pools: Connected 1, Hosts: Running 1):

1. **Click on "HOSTS" tab** to see your host details
2. **Click on your host** (`10.25.33.10`) to view its details
3. **Navigate to "Storage" section** to create the VGS ISO Storage SR
4. See the guide: `CREATE_VGS_ISO_SR_VIA_XO.md` (to be created next)

---

## Quick Reference

**Connection Details:**
- **Host**: `10.25.33.10`
- **Username**: `root`
- **Password**: `Calvin@123`
- **Port**: `443` (default)

**Where to click:**
- Top right corner: **"+ Connect pool"** button

**Success indicators:**
- Dashboard shows: "Connected 1" in Pools status
- Dashboard shows: "Running 1" in Hosts status
- HOSTS tab shows your host listed
