# Timeout Issue Analysis

## Problem Description
VM client correctly sends request to MEDIATOR, MEDIATOR processes it and sends result, but VM client times out waiting for the response.

## Analysis of Three Potential Causes

### 1. VM Initializing .txt File Before Reading Results
**Status: NOT THE ISSUE**

**Analysis:**
- VM client code (`vm_client_vector.c`) only **reads** from `response.txt` (line 174)
- VM client only **writes** to `request.txt` (line 125)
- VM never writes to or clears `response.txt`
- **Conclusion:** VM is not interfering with the response file

### 2. MEDIATOR Not Able to Write to .txt File
**Status: PARTIALLY THE ISSUE**

**Analysis:**
- MEDIATOR successfully writes to `response.txt` (lines 207-211 in `mediator_async.c`)
- However, there are **two critical problems**:
  
  **Problem A: Missing NFS Synchronization**
  - MEDIATOR writes response but does NOT use `fflush()` or `fsync()`
  - NFS may not immediately propagate the write to the VM
  - VM might poll before NFS has synchronized the file
  
  **Problem B: Immediate File Clearing (RACE CONDITION)**
  - MEDIATOR writes response at line 207-211
  - MEDIATOR **immediately clears** response file at line 228-231
  - This creates a race condition:
    - MEDIATOR writes response → closes file
    - MEDIATOR immediately re-opens in write mode (truncates to zero) → closes
    - VM polls for response → finds empty file → timeout
  
  **Conclusion:** MEDIATOR writes successfully, but clears the file too quickly, and lacks proper NFS synchronization.

### 3. Problem with VM Code
**Status: MINOR ISSUE (Can be improved)**

**Analysis:**
- VM polling logic is correct (lines 164-191)
- VM checks if file exists and has content
- However, there's a potential race condition:
  - VM opens file → checks if it has content
  - If MEDIATOR clears file between these operations, VM misses the response
- VM does NOT clear response file after reading (should do this to signal MEDIATOR that response was received)

**Conclusion:** VM code is mostly correct but could be improved to handle edge cases better.

## Root Cause Summary

**Primary Issue:** MEDIATOR clears the response file immediately after writing it (lines 228-231), creating a race condition where the VM may poll after the file has been cleared.

**Secondary Issues:**
1. Missing `fflush()` and `fsync()` after writing response (NFS synchronization)
2. VM doesn't clear response file after reading (no acknowledgment mechanism)

## Proposed Solution

### Fix 1: MEDIATOR - Remove Immediate Response Clearing
- **DO NOT** clear `response.txt` immediately after writing
- Let the VM read it first
- Clear `response.txt` only when a new request arrives from the same VM (indicating previous response was read)

### Fix 2: MEDIATOR - Add NFS Synchronization
- Add `fflush(fp)` after `fprintf()`
- Add `fsync(fileno(fp))` before `fclose()`
- Ensures NFS propagates the write before VM polls

### Fix 3: VM - Clear Response After Reading
- After successfully reading the response, clear the `response.txt` file
- This signals to MEDIATOR that the response was received
- Prevents stale responses from being read again

### Fix 4: MEDIATOR - Clear Response When New Request Arrives
- In `poll_requests()`, when detecting a new request from a VM, clear the old `response.txt` if it exists
- This ensures clean state for each new request

## Implementation Plan

1. Modify `cuda_result_callback()` in `mediator_async.c`:
   - Add `fflush()` and `fsync()` after writing response
   - Remove immediate clearing of `response.txt`
   - Keep clearing of `request.txt` (this is fine)

2. Modify `wait_for_response()` in `vm_client_vector.c`:
   - After successfully reading response, clear the `response.txt` file
   - Add proper error handling

3. Modify `poll_requests()` in `mediator_async.c`:
   - Before processing a new request, clear the old `response.txt` if it exists
   - This ensures clean state

## Expected Behavior After Fix

1. MEDIATOR writes response with proper NFS synchronization
2. VM polls and successfully reads response
3. VM clears response file after reading
4. MEDIATOR clears response file when new request arrives (if VM didn't clear it)
5. No race conditions, no timeouts
