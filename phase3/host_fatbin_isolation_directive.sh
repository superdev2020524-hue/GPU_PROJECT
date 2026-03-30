#!/usr/bin/env bash
set -euo pipefail

# Host-side directive script for isolating CUDA_ERROR_INVALID_IMAGE on
# cuModuleLoadFatBinary(data_len=401312).
#
# What this script does:
# 1) Patches src/cuda_executor.c to dump the 401312 fatbin to /tmp/fail401312.bin
# 2) Rebuilds mediator_phase3
# 3) Restarts mediator_phase3 with log at /tmp/mediator.log
# 4) Builds standalone /root/phase3/test_fatbin_load
# 5) Prints the exact post-run commands to validate root cause
#
# Expected host layout:
#   /root/phase3/{src,cuda_executor.c,mediator_phase3,Makefile}
#
# Usage:
#   bash /root/phase3/host_fatbin_isolation_directive.sh

PHASE3_DIR="${PHASE3_DIR:-/root/phase3}"
SRC_FILE="$PHASE3_DIR/src/cuda_executor.c"
TEST_SRC="$PHASE3_DIR/test_fatbin_load.c"
TEST_BIN="$PHASE3_DIR/test_fatbin_load"
MEDIATOR_BIN="$PHASE3_DIR/mediator_phase3"

if [[ ! -d "$PHASE3_DIR" ]]; then
  echo "[ERROR] Missing directory: $PHASE3_DIR"
  exit 1
fi
if [[ ! -f "$SRC_FILE" ]]; then
  echo "[ERROR] Missing source file: $SRC_FILE"
  exit 1
fi

echo "[1/5] Patching cuda_executor.c to dump failing 401312 fatbin..."
python3 - "$SRC_FILE" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
needle = "            rc = cuModuleLoadFatBinary(mod_out, fatbin_copy);\n"

if "dumped /tmp/fail401312.bin" in text:
    print("[INFO] Dump patch already present, skipping.")
    sys.exit(0)

insert = (
    "            if (data_len == 401312U) {\n"
    "                FILE *df = fopen(\"/tmp/fail401312.bin\", \"wb\");\n"
    "                if (df) {\n"
    "                    fwrite(fatbin_copy, 1, data_len, df);\n"
    "                    fclose(df);\n"
    "                    fprintf(stderr, \"[cuda-executor] dumped /tmp/fail401312.bin (%u bytes)\\n\", data_len);\n"
    "                    fflush(stderr);\n"
    "                }\n"
    "            }\n"
)

if needle not in text:
    print("[ERROR] Could not find expected insertion point in cuda_executor.c")
    sys.exit(2)

text = text.replace(needle, insert + needle, 1)
path.write_text(text)
print("[OK] Patch inserted.")
PY

echo "[2/5] Backing up patched file..."
cp -a "$SRC_FILE" "$SRC_FILE.$(date +%Y%m%d_%H%M%S).bak"

echo "[3/5] Rebuilding mediator_phase3..."
cd "$PHASE3_DIR"
make -j"$(nproc)"

if [[ ! -x "$MEDIATOR_BIN" ]]; then
  echo "[ERROR] mediator binary not found after build: $MEDIATOR_BIN"
  exit 1
fi

echo "[4/5] Restarting mediator_phase3..."
pkill -f mediator_phase3 2>/dev/null || true
sleep 1
"$MEDIATOR_BIN" 2>/tmp/mediator.log &
sleep 1
echo "[OK] mediator restarted. pid(s):"
pgrep -af mediator_phase3 || true

echo "[5/5] Building standalone fatbin loader test..."
cat > "$TEST_SRC" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <fatbin.bin>\n", argv[0]); return 2; }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 3; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc((size_t)n);
    if (!buf) return 4;
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) return 5;
    fclose(f);

    CUresult rc;
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;

    rc = cuInit(0); if (rc) { printf("cuInit rc=%d\n", rc); return 10; }
    rc = cuDeviceGet(&dev, 0); if (rc) { printf("cuDeviceGet rc=%d\n", rc); return 11; }
    rc = cuDevicePrimaryCtxRetain(&ctx, dev); if (rc) { printf("retain rc=%d\n", rc); return 12; }
    rc = cuCtxSetCurrent(ctx); if (rc) { printf("setcurrent rc=%d\n", rc); return 13; }

    rc = cuModuleLoadFatBinary(&mod, buf);
    const char *name = 0, *msg = 0;
    cuGetErrorName(rc, &name);
    cuGetErrorString(rc, &msg);
    printf("cuModuleLoadFatBinary rc=%d name=%s msg=%s size=%ld\n",
           (int)rc, name ? name : "?", msg ? msg : "?", n);

    if (rc == CUDA_SUCCESS) cuModuleUnload(mod);
    cuCtxSetCurrent(NULL);
    cuDevicePrimaryCtxRelease(dev);
    free(buf);
    return rc == CUDA_SUCCESS ? 0 : 1;
}
EOF

gcc -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
  -o "$TEST_BIN" "$TEST_SRC" -lcuda

echo
echo "=== NEXT ACTIONS (manual trigger + verification) ==="
echo "1) Trigger one VM generate run (from your normal VM path)."
echo "2) After failure/repro, run:"
echo "   ls -l /tmp/fail401312.bin"
echo "   sha256sum /tmp/fail401312.bin"
echo "   $TEST_BIN /tmp/fail401312.bin"
echo "3) Correlate logs:"
echo "   tail -n 400 /tmp/mediator.log | awk '/module-load start call_id=0x0042|module-load done call_id=0x0042|dumped \\/tmp\\/fail401312.bin|INVALID_IMAGE/'"
echo
echo "Interpretation:"
echo "  - Standalone test returns INVALID_IMAGE  => image/content compatibility issue."
echo "  - Standalone test returns CUDA_SUCCESS   => mediator call-state/context issue."
