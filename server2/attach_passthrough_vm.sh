#!/usr/bin/env bash
# =============================================================================
# attach_passthrough_vm.sh
#
# Host-side helper for Server 2 cloud deployment.
# Run this ON THE XCP-ng HOST with a VM UUID to attach the real GPU to that VM.
#
# Usage:
#   ./attach_passthrough_vm.sh <vm-uuid>
#
# Optional:
#   GPU_BDF=0000:81:00.0 ./attach_passthrough_vm.sh <vm-uuid>
# =============================================================================

set -euo pipefail

VM_UUID="${1:-}"
GPU_BDF="${GPU_BDF:-0000:81:00.0}"
GPU_KEY="0/$GPU_BDF"

info() {
    echo "[attach-passthrough] $*"
}

die() {
    echo "[attach-passthrough] ERROR: $*" >&2
    exit 1
}

get_vm_field() {
    local vm_uuid="$1"
    local field="$2"
    xe vm-list uuid="$vm_uuid" params="$field" --minimal 2>/dev/null || true
}

wait_for_vm_halt() {
    local vm_uuid="$1"
    local attempt
    local state
    for attempt in $(seq 1 30); do
        state="$(get_vm_field "$vm_uuid" power-state)"
        if [[ "$state" == "halted" ]]; then
            return 0
        fi
        sleep 2
    done
    return 1
}

pci_cfg_has_gpu() {
    local pci_cfg="$1"
    local entry
    local entries=()
    IFS=',' read -r -a entries <<< "$pci_cfg"
    for entry in "${entries[@]}"; do
        if [[ "$entry" == "$GPU_KEY" ]]; then
            return 0
        fi
    done
    return 1
}

remove_gpu_from_pci_cfg() {
    local pci_cfg="$1"
    local entry
    local kept=()
    local entries=()
    IFS=',' read -r -a entries <<< "$pci_cfg"
    for entry in "${entries[@]}"; do
        if [[ -n "$entry" && "$entry" != "$GPU_KEY" ]]; then
            kept+=("$entry")
        fi
    done
    local joined=""
    if [[ "${#kept[@]}" -gt 0 ]]; then
        printf -v joined '%s,' "${kept[@]}"
        joined="${joined%,}"
    fi
    printf '%s\n' "$joined"
}

release_gpu_from_other_vms() {
    local all_vm_uuids
    local other_uuid
    local other_name
    local other_state
    local other_pci_cfg
    local other_pci_after
    local uuid_list=()

    all_vm_uuids="$(xe vm-list params=uuid --minimal 2>/dev/null || true)"
    IFS=',' read -r -a uuid_list <<< "$all_vm_uuids"

    for other_uuid in "${uuid_list[@]}"; do
        [[ -n "$other_uuid" ]] || continue
        [[ "$other_uuid" == "$VM_UUID" ]] && continue

        other_pci_cfg="$(xe vm-param-get uuid="$other_uuid" param-name=other-config param-key=pci 2>/dev/null || true)"
        [[ -n "$other_pci_cfg" ]] || continue
        pci_cfg_has_gpu "$other_pci_cfg" || continue

        other_name="$(get_vm_field "$other_uuid" name-label)"
        other_state="$(get_vm_field "$other_uuid" power-state)"
        info "GPU is currently referenced by VM: ${other_name:-$other_uuid} ($other_uuid)"

        if [[ "$other_state" == "running" ]]; then
            info "Stopping ${other_name:-$other_uuid} to free GPU"
            xe vm-shutdown uuid="$other_uuid" || die "failed to stop ${other_name:-$other_uuid}"
            wait_for_vm_halt "$other_uuid" || die "${other_name:-$other_uuid} did not halt in time"
        fi

        other_pci_after="$(remove_gpu_from_pci_cfg "$other_pci_cfg")"
        if [[ -n "$other_pci_after" ]]; then
            xe vm-param-set uuid="$other_uuid" other-config:pci="$other_pci_after"
            info "Removed $GPU_KEY from ${other_name:-$other_uuid}; remaining pci map: $other_pci_after"
        else
            xe vm-param-remove uuid="$other_uuid" param-name=other-config param-key=pci || true
            info "Cleared passthrough GPU setting from ${other_name:-$other_uuid}"
        fi
    done
}

if [[ -z "$VM_UUID" ]]; then
    die "usage: $0 <vm-uuid>"
fi

if [[ "$(id -u)" -ne 0 ]]; then
    die "run this script as root on the host"
fi

command -v xe >/dev/null 2>&1 || die "'xe' command not found on this host"

VM_NAME="$(xe vm-list uuid="$VM_UUID" params=name-label --minimal 2>/dev/null || true)"
[[ -n "$VM_NAME" ]] || die "VM not found for uuid=$VM_UUID"

info "Target VM UUID : $VM_UUID"
info "Target VM name : $VM_NAME"
info "GPU BDF        : $GPU_BDF"

info "Step 1/6: stop target VM if it is running"
VM_POWER_STATE="$(xe vm-list uuid="$VM_UUID" params=power-state --minimal 2>/dev/null || true)"
if [[ "$VM_POWER_STATE" == "running" ]]; then
    xe vm-shutdown uuid="$VM_UUID" || true
    wait_for_vm_halt "$VM_UUID" || die "$VM_NAME did not halt in time"
else
    info "Target VM is already stopped"
fi

info "Step 2/6: stop any other VM currently using this GPU"
release_gpu_from_other_vms

info "Step 3/6: remove old custom device-model args if present"
xe vm-param-remove uuid="$VM_UUID" param-name=platform param-key=device-model-args || true

info "Step 4/6: attach the real GPU by passthrough"
xe vm-param-set uuid="$VM_UUID" other-config:pci="0/$GPU_BDF"

info "Step 5/6: keep Secure Boot disabled"
xe vm-param-set uuid="$VM_UUID" platform:secureboot=false

info "Step 6/6: start the VM"
xe vm-start uuid="$VM_UUID"

info "Done. Current passthrough-related settings:"
echo "  pci        : $(xe vm-param-get uuid="$VM_UUID" param-name=other-config param-key=pci 2>/dev/null || true)"
echo "  secureboot : $(xe vm-param-get uuid="$VM_UUID" param-name=platform param-key=secureboot 2>/dev/null || true)"
echo "  power-state: $(xe vm-list uuid="$VM_UUID" params=power-state --minimal 2>/dev/null || true)"
echo

echo "Next step:"
echo "  once the VM finishes booting, send me the VM IP address."
