# Timeout Fix Implementation Summary

## Changes Made

### 1. MEDIATOR (`mediator_async.c`) - Response Writing with NFS Synchronization

**Location:** `cuda_result_callback()` function (lines 203-215)

**Changes:**
- Added `fflush(fp)` after `fprintf()` to ensure data is written to the buffer
- Added `fsync(fileno(fp))` before `fclose()` to force NFS synchronization
- This ensures the response file is fully written and synchronized to NFS before the VM polls for it

**Before:**
```c
fprintf(fp, "%d\n", result);
fclose(fp);
```

**After:**
```c
fprintf(fp, "%d\n", result);
fflush(fp);  // Flush to ensure data is written
fsync(fileno(fp));  // Force NFS synchronization
fclose(fp);
```

### 2. MEDIATOR (`mediator_async.c`) - Removed Immediate Response Clearing

**Location:** `cuda_result_callback()` function (lines 217-231)

**Changes:**
- Removed immediate clearing of `response.txt` after writing
- Only clear `request.txt` (which is fine, as the request has been processed)
- Response file will be cleared by VM after reading, or by MEDIATOR when a new request arrives

**Before:**
```c
// Clear response file (already written, but clear for next time)
fp = fopen(response_file, "w");
if (fp) {
    fclose(fp);  // Clear after VM reads it
}
```

**After:**
- Removed entirely. Response file is now cleared by VM or when new request arrives.

### 3. MEDIATOR (`mediator_async.c`) - Clear Response When New Request Arrives

**Location:** `poll_requests()` function (lines 344-352)

**Changes:**
- Before processing a new request, clear the old `response.txt` if it exists
- This ensures clean state for each new request
- Prevents stale responses from being read

**Before:**
```c
// Construct request file path
char request_file[512];
snprintf(request_file, sizeof(request_file), "%s/%s/request.txt", NFS_BASE_DIR, entry->d_name);

// Check if file exists and is readable
FILE *fp = fopen(request_file, "r");
```

**After:**
```c
// Construct request file path
char request_file[512];
snprintf(request_file, sizeof(request_file), "%s/%s/request.txt", NFS_BASE_DIR, entry->d_name);

// Extract VM ID from directory name for response file clearing
uint32_t dir_vm_id;
if (sscanf(entry->d_name, "vm%u", &dir_vm_id) != 1) {
    continue;
}

// Clear old response file if it exists (indicates previous response was not read, or new request)
char response_file[512];
snprintf(response_file, sizeof(response_file), "%s/%s/response.txt", NFS_BASE_DIR, entry->d_name);
FILE *fp = fopen(response_file, "w");
if (fp) {
    fclose(fp);  // Clear old response
}

// Check if request file exists and is readable
fp = fopen(request_file, "r");
```

### 4. VM Client (`vm_client_vector.c`) - Clear Response After Reading

**Location:** `wait_for_response()` function (lines 173-187)

**Changes:**
- After successfully reading the response, clear the `response.txt` file
- This signals to MEDIATOR that the response was received
- Prevents stale responses from being read again

**Before:**
```c
if (sscanf(line, "%d", result) == 1) {
    fclose(fp);
    printf("[RESPONSE] Received: %d\n", *result);
    return 0;
}
```

**After:**
```c
if (sscanf(line, "%d", result) == 1) {
    fclose(fp);
    printf("[RESPONSE] Received: %d\n", *result);
    
    // Clear response file after reading to signal MEDIATOR that response was received
    fp = fopen(response_file, "w");
    if (fp) {
        fclose(fp);  // Truncate to zero
        printf("[CLEANUP] Cleared response file\n");
    }
    
    return 0;
}
```

## Expected Behavior After Fix

1. **MEDIATOR writes response:**
   - Writes result to `response.txt`
   - Flushes and syncs to ensure NFS propagation
   - Does NOT clear response file immediately

2. **VM polls for response:**
   - Polls `response.txt` file
   - Reads result when available
   - Clears response file after reading

3. **MEDIATOR detects new request:**
   - Clears old `response.txt` if it exists (cleanup for stale responses)
   - Processes new request

4. **No race conditions:**
   - Response file is available for VM to read
   - Proper NFS synchronization ensures VM sees the file
   - Clean state management prevents stale data

## Testing Recommendations

1. **Test single request:**
   - VM sends request → MEDIATOR processes → VM receives response
   - Verify no timeout

2. **Test multiple requests:**
   - Multiple VMs send requests
   - Verify all receive responses without timeout

3. **Test rapid requests:**
   - Same VM sends multiple requests quickly
   - Verify each response is received correctly

4. **Check file state:**
   - Monitor `response.txt` files in NFS directory
   - Verify they are cleared after reading
   - Verify they are cleared when new request arrives

## Files Modified

- `/home/david/Downloads/gpu/step2_test/mediator_async.c`
- `/home/david/Downloads/gpu/step2_test/vm_client_vector.c`

## Next Steps

1. Rebuild both components:
   - `make dom0` (on host)
   - `make vm` (on VMs)

2. Test the fix:
   - Start MEDIATOR on host
   - Run VM client on a VM
   - Verify no timeout occurs

3. Monitor logs:
   - Check MEDIATOR logs for response sending
   - Check VM client logs for response reception
   - Verify file clearing messages
