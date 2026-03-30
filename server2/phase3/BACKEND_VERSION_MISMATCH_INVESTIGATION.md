# Backend Version Mismatch Investigation

## Date: 2026-02-27

## ChatGPT's Hypothesis

**Ollama binary expects a different CUDA backend version than what's available.**

### Key Insight

Discovery logs show:
- Skips `cuda_v13` (expected - we requested `cuda_v12`)
- **NO attempt to load `cuda_v12`**
- `initial_count=0`

This suggests the binary expects `cuda_v13` but we only have `cuda_v12`, OR the binary wasn't built with CUDA support.

## How Ollama Backend Selection Works

Ollama has an internal backend registry compiled into the binary:

```go
var backends = []Backend{
    cpu,
    cuda_v12,
    cuda_v13,
    vulkan,
}
```

If the binary was built expecting `cuda_v13` but only `cuda_v12` exists, it will skip it silently.

## What to Check

1. **What CUDA versions does the binary expect?**
   - `strings ollama | grep cuda_v`

2. **What versions do we have?**
   - `ls /usr/local/lib/ollama/ | grep cuda`

3. **Is there a version mismatch?**
   - Binary expects `cuda_v13` but we have `cuda_v12`?

4. **Was binary built with CUDA support?**
   - `ollama env` or `ollama version`

## Possible Solutions

1. **Rename directory** - If binary expects `cuda_v13`, rename `cuda_v12` to `cuda_v13`
2. **Install correct version** - Get the backend version the binary expects
3. **Rebuild Ollama** - Build with CUDA support if missing
