#!/bin/bash
# Run on VM: during a generate, find any PID that has resource0 open.
# Usage: start generate in background, then run: sudo ./check_bar0_during_generate.sh
RES0="/sys/bus/pci/devices/0000:00:05.0/resource0"
for p in /proc/[0-9]*; do
  [ -d "$p/fd" ] || continue
  pid=${p#/proc/}
  for f in "$p"/fd/*; do
    [ -L "$f" ] || continue
    r=$(readlink "$f" 2>/dev/null)
    [ "$r" = "$RES0" ] && echo "BAR0_OPEN_PID=$pid" && break
  done
done
