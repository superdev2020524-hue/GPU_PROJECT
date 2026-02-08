# Create VGS ISO Storage SR via Xen Orchestra - Step by Step

## Current Status
✅ You have successfully:
- Connected Xen Orchestra to your XCP-ng host (10.25.33.10)
- Can see the Storage repositories page
- Can see existing SRs including the problematic SMB ISO library (red X icon)

## Goal
Create a new file-based ISO Storage Repository named "VGS ISO Storage" that points to `/mnt/iso-storage` on dom0.

## Prerequisites
- The directory `/mnt/iso-storage` must exist on dom0 and contain your ISO files
- You should already have `ubuntu-22.04.5-desktop-amd64.iso` in that directory

**Verify on dom0:**
```bash
ssh root@10.25.33.10
ls -lh /mnt/iso-storage/
# Should show your ISO file
```

---

## Step-by-Step Instructions

### Step 1: Click "New" or "+" Button

**Location**: On the Storage repositories page, look for a button to create a new SR.

**Where to find it:**
- Look at the **top right** of the Storage repositories table/area
- You might see:
  - A **"+ New"** button
  - A **"+"** icon button
  - A **"Add"** or **"Create"** button
  - Or a button with a **plus icon** (+)

**Alternative locations:**
- In the **toolbar** above the table
- In a **menu** (three dots or hamburger icon)
- As a **floating action button** (FAB) in the bottom right

**Action**: Click this button to open the "New Storage Repository" dialog.

---

### Step 2: Select Storage Type

After clicking "New", a dialog/modal should appear asking you to select the storage type.

**Look for:**
- A dropdown menu or list of storage types
- Options might include:
  - "Local Storage"
  - "NFS"
  - "SMB/CIFS"
  - "ISO library"
  - "File system (ISO library)"
  - "Other"

**Action**: Select **"ISO library"** or **"File system (ISO library)"** or **"Local ISO library"**

**Note**: If you see "SMB/CIFS" or "NFS", do NOT select those. We want a local file-based ISO library.

---

### Step 3: Fill in Storage Repository Details

After selecting the type, you'll see a form with fields to fill in.

**Fill in the following:**

1. **Name / Label**:
   - Enter: `VGS ISO Storage`
   - (This is the friendly name that will appear in the list)

2. **Path / Location / Directory**:
   - Enter: `/mnt/iso-storage`
   - (This is the directory on dom0 where your ISO files are stored)

3. **Description** (optional, if shown):
   - You can enter: `Local file-based ISO storage on VGS`
   - Or leave blank

4. **Other fields** (if any):
   - **Access mode**: Should default to "Local" (correct)
   - **Content type**: Should default to "ISO" (correct)
   - Leave other fields as default unless you know specific settings

---

### Step 4: Review and Create

**Before clicking Create:**
- Double-check the **Name**: `VGS ISO Storage`
- Double-check the **Path**: `/mnt/iso-storage`

**Action**: Click the **"Create"**, **"Add"**, **"Save"**, or **"Finish"** button at the bottom of the dialog.

---

### Step 5: Wait for Creation

Xen Orchestra will create the SR. This may take 10-30 seconds.

**What to expect:**
- A loading spinner or progress indicator
- The dialog may close automatically on success
- You may see a success notification
- The page may refresh or the new SR may appear in the list

---

### Step 6: Verify SR Was Created

**Check the Storage repositories table:**
- Look for **"VGS ISO Storage"** in the list
- It should show:
  - **STORAGE REPOSITORY**: "VGS ISO Storage"
  - **DESCRIPTION**: Something like "File system (ISO library)" or "Local ISO"
  - **STORAGE FORMAT**: "iso" or "file"
  - **ACCESS MODE**: "Local"
  - **# USED SPACE**: Should show some space if ISO files are detected

**Check for errors:**
- If there's a **red X icon** next to the name, there's a problem (see troubleshooting)
- If the SR appears but shows "0" used space, it might not have scanned yet

---

### Step 7: Verify ISO Files Are Visible

**Click on "VGS ISO Storage"** in the table to view its details.

**What you should see:**
- Details panel on the right should show SR information
- You might see a list of ISO files, or a button to view files
- Look for your Ubuntu ISO: `ubuntu-22.04.5-desktop-amd64.iso`

**If ISO files don't appear:**
- The SR might need to be scanned
- See troubleshooting section below

---

## Troubleshooting

### Problem: No "New" or "+" button visible

**Solution 1: Check permissions**
- Make sure you're logged in as an admin user
- Try logging out and back in

**Solution 2: Look in different location**
- Check if there's a **menu** (three dots) on the Storage page
- Look for **"Actions"** or **"More"** menu
- Check the **top navigation bar** for a global "New" button

**Solution 3: Try alternative path**
- Click on your **pool name** ("xcp-ng-syovfxoz") in the left sidebar
- Look for Storage section there
- Or navigate: **SYSTEM** tab → **Storage** section

---

### Problem: "ISO library" option not available

**Solution:**
- Make sure you're selecting the correct type
- Some versions call it "File system (ISO library)" or "Local ISO library"
- If you only see "SMB/CIFS" and "NFS", you might need to use CLI method instead
- Try refreshing the page and trying again

---

### Problem: SR created but shows red X icon (error)

**Check 1: Path exists on dom0**
```bash
# SSH to dom0
ssh root@10.25.33.10
ls -ld /mnt/iso-storage
# Should show the directory exists
```

**Check 2: Path is readable**
```bash
# On dom0
ls -la /mnt/iso-storage/
# Should list ISO files
```

**Check 3: Permissions**
```bash
# On dom0
stat /mnt/iso-storage
# Should show readable permissions
```

**Check 4: XAPI can access it**
```bash
# On dom0
xe sr-list name-label="VGS ISO Storage" params=uuid,name-label,type
xe pbd-list sr-uuid=<SR_UUID> params=uuid,currently-attached
```

**Solution:**
- If path doesn't exist, create it: `mkdir -p /mnt/iso-storage`
- If permissions are wrong, fix them: `chmod 755 /mnt/iso-storage`
- If XAPI can't attach, try: `xe pbd-plug uuid=<PBD_UUID>`

---

### Problem: SR created but shows "0" used space / no ISO files

**Solution 1: Scan the SR**
- Click on the SR in the table
- Look for a **"Scan"** or **"Refresh"** button
- Click it to force XO to rescan for ISO files

**Solution 2: Verify files exist**
```bash
# On dom0
ls -lh /mnt/iso-storage/
# Should show your ISO file(s)
```

**Solution 3: Check SR type**
- Make sure the SR type is "ISO library" not "Local storage"
- ISO libraries are specifically for ISO files

---

### Problem: "Path does not exist" or "Permission denied" error

**Solution:**
```bash
# On dom0, create and set permissions
ssh root@10.25.33.10
mkdir -p /mnt/iso-storage
chmod 755 /mnt/iso-storage
ls -la /mnt/iso-storage/
# Verify ISO files are there
```

Then try creating the SR again in XO.

---

### Problem: SR appears but VM creation still fails

**Verify SR is actually attached:**
```bash
# On dom0
xe sr-list name-label="VGS ISO Storage" params=uuid
SR_UUID=<paste_uuid_here>
xe pbd-list sr-uuid=$SR_UUID params=uuid,currently-attached
```

**If PBD shows `currently-attached: false`:**
```bash
PBD_UUID=<paste_pbd_uuid_here>
xe pbd-plug uuid=$PBD_UUID
```

**Then verify SR is accessible:**
```bash
SR_MOUNT="/var/run/sr-mount/$SR_UUID"
ls -la "$SR_MOUNT"
# Should list ISO files
```

---

## Alternative: Create via CLI (If XO Method Fails)

If you cannot create the SR via Xen Orchestra web interface, you can create it via CLI:

```bash
# SSH to dom0
ssh root@10.25.33.10

# Create the SR
xe sr-create \
  name-label="VGS ISO Storage" \
  type=iso \
  content-type=iso \
  device-config:location=/mnt/iso-storage \
  device-config:legacy_mode=true \
  shared=false

# Get the SR UUID
SR_UUID=$(xe sr-list name-label="VGS ISO Storage" params=uuid --minimal)

# Plug the PBD (attach the SR)
PBD_UUID=$(xe pbd-list sr-uuid=$SR_UUID params=uuid --minimal)
xe pbd-plug uuid=$PBD_UUID

# Scan for ISO files
xe sr-scan uuid=$SR_UUID

# Verify
xe sr-list name-label="VGS ISO Storage" params=uuid,name-label,type,content-type
ls -la /var/run/sr-mount/$SR_UUID/
```

---

## Next Steps After Successful Creation

Once "VGS ISO Storage" appears in the list **without a red X icon**:

1. **Verify ISO files are visible** (click on the SR to see details)
2. **Run the VM creation script**:
   ```bash
   bash /home/david/Downloads/gpu/vm_create/create_test3_vm.sh
   ```
3. The script should now find "VGS ISO Storage" and proceed successfully!

---

## Quick Reference

**SR Details:**
- **Name**: `VGS ISO Storage`
- **Type**: ISO library / File system (ISO library)
- **Path**: `/mnt/iso-storage`
- **Access Mode**: Local
- **Content Type**: ISO

**Where to click:**
- Storage repositories page → **"+ New"** or **"+"** button
- Select **"ISO library"** type
- Fill in Name and Path
- Click **"Create"**

**Success indicators:**
- SR appears in list without red X icon
- Shows used space > 0 (if ISO files are present)
- ISO files are visible when clicking on the SR
