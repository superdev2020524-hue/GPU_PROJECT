#!/bin/bash
# Fix vgpu.conf syntax issues

set -e

CONF_FILE="/etc/systemd/system/ollama.service.d/vgpu.conf"

echo "Fixing vgpu.conf syntax issues..."

# Backup
sudo cp "$CONF_FILE" "${CONF_FILE}.backup.$(date +%Y%m%d_%H%M%S)"

# Remove duplicate OLLAMA_LLM_LIBRARY lines (without quotes)
sudo sed -i '/^Environment=OLLAMA_LLM_LIBRARY=cuda_v12$/d' "$CONF_FILE"

# Add correct line with quotes if it doesn't exist
if ! grep -q 'Environment="OLLAMA_LLM_LIBRARY=cuda_v12"' "$CONF_FILE"; then
    echo 'Environment="OLLAMA_LLM_LIBRARY=cuda_v12"' | sudo tee -a "$CONF_FILE" > /dev/null
fi

# Verify
echo ""
echo "Verification:"
sudo grep "OLLAMA" "$CONF_FILE"

echo ""
echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "Restarting Ollama..."
sudo systemctl restart ollama

echo ""
echo "Waiting 20 seconds for Ollama to start..."
sleep 20

echo ""
echo "Checking service status:"
sudo systemctl is-active ollama

echo ""
echo "Checking GPU mode (last 30 seconds):"
sudo journalctl -u ollama --since "30 seconds ago" --no-pager 2>&1 | grep -E "initial_count|library=|pci_id" | tail -5

echo ""
echo "Fix complete!"
