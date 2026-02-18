# Phase 3: Scheduler Upgrade & Isolation Controls

## Overview

Phase 3 extends our existing XCP-ng vGPU solution with scheduling and isolation features. We're building on what we already have working: the MMIO-based vGPU stub device that integrates with Xen's QEMU device model, the Unix socket mediator daemon running on the Dom0 host, and the SQLite database that tracks VM-to-pool assignments.

The goal here is to make GPU sharing across multiple VMs both fair and safe. Right now, Phase 2 gives us basic round-robin scheduling and pool isolation. Phase 3 adds:

**Weighted Fair Queuing Scheduler** — Instead of simple round-robin, we're implementing a demand-aware scheduler that considers VM priority, admin-assigned weights, current queue depth, and wait time. This means VMs with higher weights or more urgent workloads get proportionally more GPU time, while still preventing starvation of lower-priority VMs.

**Per-VM Rate Limiting** — Each VM gets configurable limits on how many jobs it can submit per second and how many can be queued at once. When a VM hits these limits, requests are rejected with a BUSY signal that the guest client handles with automatic retry. This prevents any single VM from flooding the queue or monopolizing the GPU.

**Watchdog & Auto-Quarantine** — A background thread monitors job execution times. If a VM's jobs consistently timeout or fail, it gets automatically quarantined. Quarantined VMs can't submit new jobs until an admin clears the flag. The watchdog also tracks GPU health via NVML and can trigger resets when needed.

**Metrics & Observability** — We're collecting latency percentiles (p50, p95, p99), queue depths, context switch counts, and error rates. This data is available through the admin CLI in both human-readable format and Prometheus export, so you can integrate it with your existing monitoring stack.

**NVML Integration** — GPU health monitoring (temperature, ECC errors, power draw, utilization) is loaded at runtime, so the system works even if NVML isn't installed. If it's available, the mediator uses it to feed the watchdog and provide health status through the admin interface.

The communication path stays the same as Phase 2: guest VMs write to MMIO registers in the vGPU stub device (exposed via PCI BAR0), the stub forwards requests over a Unix domain socket to the mediator daemon in Dom0, and responses flow back the same way. No changes to the protocol wire format or the QEMU integration points.

## Implementation Steps

### Step 1: Database Schema Extension

The configuration database is extended to store new per-VM settings for Phase 3 features. Each VM can now have:
- A scheduler weight value (1.0 to 100.0) that determines its share of GPU time relative to other VMs
- Rate limit settings (maximum jobs per second)
- Queue depth limits (maximum number of jobs that can be queued for this VM)
- Quarantine status (automatically set by the watchdog when a VM misbehaves)
- Error counters (tracked by the watchdog for fault detection)

Existing Phase 2 databases are automatically upgraded when the mediator starts, so no manual migration is required. Fresh installations include these fields from the start.

**Result:** Administrators can configure per-VM scheduling weights, rate limits, and queue depths through the admin CLI. The system tracks VM health and quarantine status automatically.

### Step 2: Protocol Extensions

The communication protocol between guest VMs and the mediator is extended to support new error conditions. VMs can now receive specific error codes indicating:
- Rate limiting (the VM has exceeded its configured submission rate or queue depth)
- Quarantine status (the VM has been automatically disabled due to repeated failures)

Additionally, a new admin command channel is established between the CLI tool and the running mediator daemon, allowing administrators to query metrics and health status in real-time without restarting services.

**Result:** Guest applications receive clear error signals when they're rate-limited or quarantined, so they can implement retry logic. Administrators can query system status and metrics through the CLI without disrupting operations.

### Step 3: Weighted Fair Queuing Scheduler

The scheduler is upgraded from simple round-robin to a weighted fair queuing algorithm. Each GPU job request is assigned an urgency score based on:
- The VM's priority level (high, medium, or low)
- The VM's admin-assigned weight
- How many jobs from that VM are already queued (demand pressure)
- How long the request has been waiting (prevents starvation)

The scheduler maintains real-time statistics for each VM, tracking queue depth, submission rates, and average execution times. This data is used both for scheduling decisions and for metrics reporting.

**Result:** VMs with higher weights or more urgent workloads receive proportionally more GPU time, while lower-priority VMs are still guaranteed fair access. The system adjusts based on current queue state and wait times.

### Step 4: Per-VM Rate Limiter

A token-bucket rate limiter is added to enforce per-VM submission limits. Each VM has its own rate limit bucket with:
- A configurable refill rate (determines maximum jobs per second)
- A maximum bucket size (determines maximum queue depth)

When a VM attempts to submit a job that would exceed either limit, the request is immediately rejected. The guest-side client application automatically retries with exponential backoff (up to 5 attempts over a period of 100ms to 5 seconds).

**Result:** Administrators can set per-VM rate limits to prevent any single VM from monopolizing the GPU or flooding the submission queue. Guest applications retry automatically when rate-limited.

### Step 5: Watchdog & Error Recovery

A background monitoring system monitors GPU job execution. The watchdog:
- Monitors active jobs for timeouts (default 30 seconds per job)
- Tracks per-VM failure counts in the database
- Automatically quarantines VMs that exceed a configurable fault threshold
- Interfaces with NVML to detect GPU conditions that may require a reset

When a VM is quarantined, all future job submissions from that VM are immediately rejected until an administrator manually clears the quarantine flag. The guest application receives a non-retryable error code, so it knows to stop attempting submissions.

**Result:** Misbehaving VMs are automatically isolated before they can destabilize the GPU for other VMs. Administrators receive clear visibility into VM health and can manually quarantine or unquarantine VMs as needed.

### Step 6: Metrics & Observability

Metrics collection tracks performance data for every GPU operation:
- Job completion latencies with percentile calculations (p50, p95, p99)
- Per-VM queue depths
- Context switch counts (how often the scheduler switches between VMs)
- Error and rejection rates

This data is exposed through the admin CLI in two formats:
- Human-readable summary for quick status checks
- Prometheus exposition format for integration with existing monitoring infrastructure (Grafana, Prometheus, etc.)

**Result:** Administrators have visibility into GPU performance, queue depths, and system health. Metrics can be integrated into existing monitoring dashboards for alerting and capacity planning.

### Step 7: GPU Health Monitoring (NVML)

GPU health monitoring is integrated using NVIDIA's Management Library (NVML). The system polls:
- GPU temperature
- Compute and memory utilization percentages
- Memory usage (used vs. total)
- Power draw
- ECC error counts

This monitoring is loaded at runtime, so the system works even if NVML isn't installed (it simply operates without health monitoring). When available, the health data feeds into the watchdog's reset-detection logic and is exposed through the admin CLI.

**Result:** Administrators can monitor GPU health in real-time, including temperature, utilization, and error conditions. The watchdog uses this data to detect conditions that may require GPU resets.

### Step 8: Admin CLI Extensions

The `vgpu-admin` command-line tool is extended with new commands for Phase 3 features. All commands that target a specific VM accept either the VM's UUID or its name (resolved automatically via XCP-ng's `xe` toolstack), consistent with Phase 2.

New administrative capabilities:
- **Scheduler weight management:** Set and view per-VM scheduler weights
- **Rate limit configuration:** Set and view per-VM rate limits and queue depth limits
- **Quarantine management:** Manually quarantine or unquarantine VMs
- **Metrics viewing:** Display performance metrics in human-readable or Prometheus format
- **Health status:** Display real-time GPU health information
- **Config reload:** Force the mediator to reload VM configurations from the database without restarting

**Result:** Administrators have full control over scheduling policies, rate limits, and VM quarantine status through the CLI. All operations can be performed without service restarts.

### Step 9: QEMU vGPU-Stub & Guest Client Updates

The QEMU vGPU-stub device is updated to correctly interpret the new error signals from the mediator and expose them to guest VMs through MMIO registers. Guest applications can now detect:
- Rate limiting conditions (with automatic retry support)
- Quarantine status (non-retryable, requires admin intervention)

The guest-side client application is updated with retry logic:
- Rate-limit errors trigger automatic retries with exponential backoff
- Quarantine errors are immediately reported to the application (no retries)

**Result:** Guest applications receive error codes and retry automatically when rate-limited. Quarantine conditions are immediately visible to applications, so they don't waste time retrying.

### Step 10: Integration, Build System & Testing

All components are integrated and tested together. Stress tests cover:
- Baseline throughput under normal load
- Multi-VM fairness when using weighted scheduling
- Rate limiting and back-pressure behavior
- Watchdog quarantine functionality
- Metrics collection accuracy

**Result:** All components integrated and tested. Components work together, and tests validate the Phase 3 features.

## Deliverables Summary

**Software Components:**
- Enhanced mediator daemon with WFQ scheduler, rate limiting, watchdog, and metrics
- Extended admin CLI tool with Phase 3 management commands
- Updated guest client application with retry logic
- Updated QEMU vGPU-stub device with new error code support

**Administrative Capabilities:**
- Per-VM scheduler weight configuration
- Per-VM rate limit and queue depth configuration
- VM quarantine management (automatic and manual)
- Real-time metrics viewing (human-readable and Prometheus)
- GPU health monitoring and status reporting
- Configuration reload without service restart

**Operational Benefits:**
- Fair GPU resource allocation across multiple VMs
- Protection against VM misbehavior and resource monopolization
- Automatic fault detection and VM isolation
- Observability for performance tuning and troubleshooting
- Integration with existing monitoring infrastructure

**Documentation:**
- This implementation plan
- Technical implementation guide for engineering teams

## Prerequisites

- Working Phase 2 deployment:
  - MMIO vGPU-stub device integrated into QEMU
  - Mediator daemon running in Dom0
  - SQLite database with pools and VMs registered
  - Admin CLI tool functional
- CUDA Toolkit on the XCP-ng host
- Standard Linux build tools
- NVIDIA GPU driver with NVML support (optional, for health monitoring)

## Timeline

[To be determined by project management]
