# SCP Issue Resolution

## Date: 2026-02-26

## Problem

SCP was failing with errors:
```
ssh_askpass: exec(/usr/bin/ssh-askpass): No such file or directory
test-10@10.25.33.110: Permission denied (publickey,password).
scp: Connection closed
```

**Root Cause:** SCP cannot handle password authentication interactively when `ssh-askpass` is not available.

## Solution

**Base64 encoding via SSH** - Bypasses SCP entirely by:
1. Encoding file to base64 locally
2. Transferring base64 string via SSH heredoc
3. Decoding base64 on VM using Python
4. Writing decoded content to target file

### Implementation

```python
import base64

# Read file locally
with open('file.c', 'rb') as f:
    file_content = f.read()

# Encode to base64
file_b64 = base64.b64encode(file_content).decode('utf-8')

# Transfer via SSH
ssh.sendline('cat > /tmp/file.b64 << "B64END"')
# Send base64 in chunks
ssh.sendline(file_b64)
ssh.sendline('B64END')

# Decode on VM
ssh.sendline('python3 -c "import base64; f=open(\"/tmp/file.b64\"); d=f.read(); f.close(); c=base64.b64decode(d); open(\"file.c\", \"wb\").write(c)"')
```

## Status

✓ **RESOLVED** - Base64 transfer method works reliably
✓ File transfers now work without SCP
✓ Constructor fix successfully deployed using this method

## Alternative Methods

If base64 transfer has issues with large files:
1. **rsync** with password (if available)
2. **SFTP** with password handling
3. **SSH with tar pipe** (tar | ssh | tar)
4. **Manual file editing** on VM
