#!/bin/bash
# Enable coredumps for the ollama service on the VM so that the next runner
# crash produces a core file for gdb backtrace.
# Run on the VM (e.g. via: python3 connect_vm.py "bash -s" < enable_coredump_ollama_vm.sh)
# Or copy to VM and run: sudo bash enable_coredump_ollama_vm.sh

set -e
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/coredump.conf << 'EOF'
[Service]
LimitCORE=infinity
EOF
systemctl daemon-reload
systemctl restart ollama
echo "Coredumps enabled for ollama. Next crash will produce a core (see coredumpctl or core_pattern)."
