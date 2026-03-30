# How `server2/phase3/` was produced

Full **Phase 3** tree (sources, `Makefile`, `guest-shim/`, `include/`, `GOAL/`, `tests/`, scripts, and documentation) is mirrored here for **server2** deployment and offline reference.

## Sync command (re-run from repo root)

```bash
rsync -a \
  --exclude='out/' \
  --exclude='ollama-src/' \
  --exclude='ollama-src-phase3/' \
  --exclude='phase3_guest.tar' \
  --exclude='phase3.tar.gz' \
  --exclude='*.nohup.out' \
  --exclude='.git/' \
  phase3/ server2/phase3/
```

**Excluded:** large Ollama clones, build artifacts (`out/`), guest tarballs, stray nohup logs.

After pulling new Phase 3 changes, re-run this to refresh `server2/phase3/`.
