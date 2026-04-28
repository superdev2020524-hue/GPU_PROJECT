# Deployment Guide for Enhanced Error Capture System

## Overview

This guide provides step-by-step instructions to deploy the enhanced error capture system to the VM and capture errors during Ollama's GPU discovery.

## Prerequisites

- Access to VM: `test-10@10.25.33.110`
- Password: `Calvin@123`
- Local files ready in: `phase3/guest-shim/`

## Step 1: Copy Updated Files to VM

### Copy Enhanced Source File

```bash
cd /home/david/Downloads/gpu
scp phase3/guest-shim/libvgpu_cuda.c test-10@10.25.33.110:~/phase3/guest-shim/
```

### Copy Scripts

```bash
scp phase3/guest-shim/capture_errors.sh test-10@10.25.33.110:~/phase3/guest-shim/
scp phase3/guest-shim/analyze_errors.sh test-10@10.25.33.110:~/phase3/guest-shim/
scp phase3/guest-shim/verify_symbols.sh test-10@10.25.33.110:~/phase3/guest-shim/
```

### Copy Install Script (if needed)

```bash
scp phase3/guest-shim/install.sh test-10@10.25.33.110:~/phase3/guest-shim/
```

## Step 2: SSH to VM

```bash
ssh test-10@10.25.33.110
# Password: Calvin@123
```

## Step 3: Rebuild Shim Libraries

```bash
cd ~/phase3/guest-shim
chmod +x install.sh capture_errors.sh analyze_errors.sh verify_symbols.sh
sudo ./install.sh
```

This will:
- Rebuild `libvgpu-cuda.so` with enhanced write() interceptor
- Rebuild `libvgpu-cudart.so` with all Runtime API functions
- Install libraries to `/usr/lib64/`
- Update systemd configuration with stderr redirection
- Create symlinks in Ollama directories

## Step 4: Restart Ollama Service

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo systemctl status ollama
```

## Step 5: Capture Errors

### Option A: Automated Capture (Recommended)

```bash
cd ~/phase3/guest-shim
./capture_errors.sh 60
```

This will:
- Clean old log files
- Start strace on Ollama process
- Capture for 60 seconds
- Collect all error logs
- Create timestamped capture directory

### Option B: Manual Capture

```bash
# Clean old logs
rm -f /tmp/ollama_errors*.log /tmp/ollama_stderr.log

# Restart Ollama to trigger discovery
sudo systemctl restart ollama

# Wait 60 seconds for discovery to complete/timeout
sleep 60

# Check captured logs
ls -lh /tmp/ollama_errors*.log /tmp/ollama_stderr.log
cat /tmp/ollama_errors_filtered.log
```

## Step 6: Analyze Captured Errors

```bash
cd ~/phase3/guest-shim

# Find latest capture directory
CAPTURE_DIR=$(ls -td /tmp/ollama_error_capture_* | head -1)

# Analyze errors
./analyze_errors.sh "$CAPTURE_DIR"
```

This will:
- Extract unique error messages
- Extract full (non-truncated) error messages
- Categorize errors by type
- Generate comprehensive report

## Step 7: Verify Symbols (Optional)

```bash
cd ~/phase3/guest-shim
./verify_symbols.sh
```

This will:
- Check which of 39 "undefined" symbols exist
- Verify version symbols are exported correctly
- Generate verification report

## Step 8: Review Results

### Check Analysis Report

```bash
CAPTURE_DIR=$(ls -td /tmp/ollama_error_capture_* | head -1)
cat "$CAPTURE_DIR/analysis/REPORT.txt"
cat "$CAPTURE_DIR/analysis/full_error_messages.txt"
cat "$CAPTURE_DIR/analysis/unique_errors.txt"
```

### Check Individual Log Files

```bash
# Full error log (all stderr writes)
cat /tmp/ollama_errors_full.log | tail -50

# Filtered error log (errors only)
cat /tmp/ollama_errors_filtered.log

# Systemd stderr log
cat /tmp/ollama_stderr.log | tail -50

# Journalctl logs
journalctl -u ollama --since "5 minutes ago" --no-pager | tail -50
```

## Step 9: Research Solutions

Use the captured error messages to search online for solutions:

1. Copy unique error messages from `analysis/unique_errors.txt`
2. Search for each error message online
3. Look for:
   - Exact error text matches
   - Similar error patterns
   - GitHub issues
   - CUDA documentation
   - Solution workarounds

## Step 10: Implement Fixes

Based on the analysis:

1. **If missing functions**: Add to `libvgpu_cudart.c`
2. **If version symbols**: Verify version script is applied
3. **If initialization errors**: Fix in `libvgpu_cuda.c` or `libvgpu_cudart.c`
4. **If symbol resolution**: Check symlinks and LD_LIBRARY_PATH

## Step 11: Rebuild and Test

```bash
cd ~/phase3/guest-shim
sudo ./install.sh
sudo systemctl restart ollama
./capture_errors.sh 60
./analyze_errors.sh $(ls -td /tmp/ollama_error_capture_* | head -1)
```

## Troubleshooting

### Scripts Not Found

If scripts are not found, copy them manually:

```bash
# On local machine
cd /home/david/Downloads/gpu/phase3/guest-shim
scp *.sh test-10@10.25.33.110:~/phase3/guest-shim/
```

### Permission Denied

```bash
chmod +x *.sh
```

### Sudo Password Required

The scripts use `sudo` - you'll need to enter the password when prompted.

### No Errors Captured

If no errors are captured:

1. Check if write() interceptor is working:
   ```bash
   grep "CAPTURED" /tmp/ollama_errors_full.log
   ```

2. Check if Ollama is running:
   ```bash
   ps aux | grep ollama
   ```

3. Check systemd logs:
   ```bash
   journalctl -u ollama --since "5 minutes ago" --no-pager
   ```

### Strace Fails

If strace fails due to permissions:

```bash
# Run capture script with sudo
sudo ./capture_errors.sh 60
```

## Expected Output

After successful capture, you should see:

```
/tmp/ollama_error_capture_YYYYMMDD_HHMMSS/
├── errors_full.log          # All stderr writes
├── errors_filtered.log      # Filtered errors
├── stderr.log               # Systemd stderr
├── strace.log               # Strace output
├── journalctl.log            # System logs
├── processes.txt             # Process info
├── memory_maps.txt           # Memory maps
├── SUMMARY.txt               # Summary
└── analysis/
    ├── REPORT.txt            # Analysis report
    ├── unique_errors.txt     # Unique errors
    ├── full_error_messages.txt  # Full messages
    ├── error_categories.txt  # Categorized errors
    └── strace_errors.txt     # Strace errors
```

## Next Steps After Capture

1. **Review full error messages** - No more truncation!
2. **Identify root causes** - Use analysis report
3. **Research solutions** - Search online for each error
4. **Implement fixes** - Based on research
5. **Verify fixes** - Re-run capture to confirm

## Quick Reference

```bash
# Deploy
scp phase3/guest-shim/libvgpu_cuda.c *.sh test-10@10.25.33.110:~/phase3/guest-shim/

# Rebuild
ssh test-10@10.25.33.110 'cd ~/phase3/guest-shim && sudo ./install.sh'

# Capture
ssh test-10@10.25.33.110 'cd ~/phase3/guest-shim && ./capture_errors.sh 60'

# Analyze
ssh test-10@10.25.33.110 'cd ~/phase3/guest-shim && ./analyze_errors.sh $(ls -td /tmp/ollama_error_capture_* | head -1)'

# Retrieve results
scp test-10@10.25.33.110:/tmp/ollama_error_capture_* ./
```
