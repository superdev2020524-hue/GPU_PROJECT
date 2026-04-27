# VM GPU Passthrough Handoff

The passthrough script is already on the host at:

```bash
~/attach_passthrough_vm.sh
```

## 1. Get the VM UUID

If you know the VM name:

```bash
xe vm-list name-label="<vm-name>" params=uuid,name-label,power-state
```

If you need to list all VMs:

```bash
xe vm-list params=uuid,name-label,power-state
```

Use the `uuid` value of the VM that should receive the GPU.

## 2. Run the passthrough script

On the host:

```bash
chmod +x ~/attach_passthrough_vm.sh
~/attach_passthrough_vm.sh <vm-uuid>
```

The script will:

- stop the target VM if it is already running
- stop any other VM that is currently using the GPU
- move the GPU to the target VM
- keep Secure Boot disabled on the target VM
- start the target VM

## 3. Send me the VM IP address

After the VM finishes booting, send me the VM IP address.

## If the script fails

Send me:

- the full terminal output
- the VM UUID
