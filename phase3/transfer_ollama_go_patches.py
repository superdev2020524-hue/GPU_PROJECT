#!/usr/bin/env python3
"""
Apply vGPU patches in memory, transfer patched device.go and server.go to the VM,
then build ollama.bin and install. Uses chunked base64 transfer (no full SCP of phase3).

Usage: python3 transfer_ollama_go_patches.py [REMOTE_OLLAMA]
  REMOTE_OLLAMA defaults to /home/<VM_USER>/ollama (from vm_config).
"""
import os
import sys
import base64
import hashlib
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_HOME

CHUNK_SIZE = 40000
OLLAMA_SRC = os.path.join(SCRIPT_DIR, "ollama-src")
_remote = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
REMOTE_OLLAMA = (_remote or os.path.join(REMOTE_HOME, "ollama")).rstrip("/")


def run_vm(cmd, timeout_sec=180):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout_sec, cwd=SCRIPT_DIR,
    )
    return result.returncode == 0, result.stdout or "", result.stderr or ""


def patch_device_go(content: str, require: bool = True) -> str:
    old = 'return d.Library == "ROCm" || d.Library == "CUDA"'
    new = 'return d.Library == "ROCm"'
    if new in content and old not in content:
        return content  # already patched
    if old not in content:
        if require:
            raise SystemExit("device.go: target line not found")
        return content
    return content.replace(old, new, 1)


def patch_server_go(content: str) -> str:
    # 1) Prepend /opt/vgpu/lib to libraryPaths
    needle1 = "\tlibraryPaths := append([]string{}, gpuLibs...)\n\tif libraryPath"
    insert1 = (
        "\tlibraryPaths := append([]string{}, gpuLibs...)\n"
        "\t// vGPU guest: ensure runner loads shims from /opt/vgpu/lib first\n"
        '\tif len(libraryPaths) == 0 || libraryPaths[0] != "/opt/vgpu/lib" {\n'
        '\t\tlibraryPaths = append([]string{"/opt/vgpu/lib"}, libraryPaths...)\n'
        "\t}\n"
        "\tif libraryPath"
    )
    if insert1 not in content and needle1 in content:
        content = content.replace(needle1, insert1, 1)

    # 2) OLLAMA_LIBRARY_PATH prepend
    old_ollama = 'cmd.Env[i] = "OLLAMA_LIBRARY_PATH=" + strings.Join(gpuLibs, string(filepath.ListSeparator))'
    new_ollama = 'cmd.Env[i] = "OLLAMA_LIBRARY_PATH=" + "/opt/vgpu/lib:" + strings.Join(gpuLibs, string(filepath.ListSeparator))'
    if old_ollama in content and new_ollama not in content:
        content = content.replace(old_ollama, new_ollama, 1)
    old_append = 'cmd.Env = append(cmd.Env, "OLLAMA_LIBRARY_PATH="+strings.Join(gpuLibs, string(filepath.ListSeparator)))'
    new_append = 'cmd.Env = append(cmd.Env, "OLLAMA_LIBRARY_PATH="+"/opt/vgpu/lib:"+strings.Join(gpuLibs, string(filepath.ListSeparator)))'
    if old_append in content and new_append not in content:
        content = content.replace(old_append, new_append, 1)

    # 3) Runner gets LD_LIBRARY_PATH and OLLAMA_* but NOT LD_PRELOAD (VM_TEST3_GPU_MODE_STATUS.md:
    #    with LD_PRELOAD the runner dlopen never loads libggml-cuda.so; without it, dlopen loads
    #    libggml-cuda.so which pulls in our shim via LD_LIBRARY_PATH and GPU is detected)
    needle3 = (
        "\tfor k, done := range extraEnvsDone {\n\t\tif !done {\n\t\t\tcmd.Env = append(cmd.Env, k+\"=\"+extraEnvs[k])\n\t\t}\n\t}\n\n"
        "\tslog.Info(\"starting runner\""
    )
    insert3 = (
        "\tfor k, done := range extraEnvsDone {\n\t\tif !done {\n\t\t\tcmd.Env = append(cmd.Env, k+\"=\"+extraEnvs[k])\n\t\t}\n\t}\n\n"
        "\t// Ensure runner gets LD_LIBRARY_PATH, OLLAMA_* (do NOT pass LD_PRELOAD: need real dlopen to load libggml-cuda.so)\n"
        "\tfor _, key := range []string{\"LD_LIBRARY_PATH\", \"OLLAMA_LIBRARY_PATH\", \"OLLAMA_LLM_LIBRARY\", \"OLLAMA_NUM_GPU\"} {\n"
        "\t\tif v, ok := os.LookupEnv(key); ok && v != \"\" {\n"
        "\t\t\tfound := false\n"
        "\t\t\tfor i := range cmd.Env {\n"
        "\t\t\t\tif strings.HasPrefix(cmd.Env[i], key+\"=\") {\n"
        "\t\t\t\t\tcmd.Env[i] = key + \"=\" + v\n"
        "\t\t\t\t\tfound = true\n"
        "\t\t\t\t\tbreak\n"
        "\t\t\t\t}\n"
        "\t\t\t}\n"
        "\t\t\tif !found {\n"
        "\t\t\t\tcmd.Env = append(cmd.Env, key+\"=\"+v)\n"
        "\t\t\t}\n"
        "\t\t}\n"
        "\t}\n"
        "\t// Remove LD_PRELOAD from runner so dlopen loads libggml-cuda.so\n"
        "\tn := 0\n"
        "\tfor _, e := range cmd.Env {\n"
        "\t\tif !strings.HasPrefix(e, \"LD_PRELOAD=\") {\n"
        "\t\t\tcmd.Env[n] = e\n"
        "\t\t\tn++\n"
        "\t\t}\n"
        "\t}\n"
        "\tcmd.Env = cmd.Env[:n]\n\n"
        "\tslog.Info(\"starting runner\""
    )
    if insert3 not in content and needle3 in content:
        content = content.replace(needle3, insert3, 1)
    elif "Ensure runner gets LD_LIBRARY_PATH" not in content and "Remove LD_PRELOAD from runner" not in content and needle3 not in content:
        raise SystemExit("server.go: StartRunner env block target not found")

    return content


def patch_discover_runner_go(content: str) -> str:
    # Put GPU lib dir (cuda_v12) before LibOllamaPath so backend loader finds CUDA before CPU
    old = "\t\t\tdirs = []string{ml.LibOllamaPath, dir}"
    new = "\t\t\tdirs = []string{dir, ml.LibOllamaPath}"
    if old in content:
        content = content.replace(old, new, 1)
    # Refresh path: during model load GPUDevices() re-runs bootstrap with same dir order (see DISCOVER_REFRESH_CUDA.md)
    old_refresh = "updatedDevices := bootstrapDevices(ctx, []string{ml.LibOllamaPath, dir}, devFilter)"
    new_refresh = "updatedDevices := bootstrapDevices(ctx, []string{dir, ml.LibOllamaPath}, devFilter)"
    if old_refresh in content:
        content = content.replace(old_refresh, new_refresh, 1)
    return content


def transfer_file_to_vm(data: bytes, remote_path: str, label: str) -> bool:
    local_sha = hashlib.sha256(data).hexdigest()
    b64 = base64.b64encode(data).decode("ascii")

    def escape(s):
        return s.replace("'", "'\"'\"'")

    ok, _, _ = run_vm(f"rm -f /tmp/combined.b64")
    for i in range(0, len(b64), CHUNK_SIZE):
        chunk = b64[i : i + CHUNK_SIZE]
        cmd = "echo -n '" + escape(chunk) + "' >> /tmp/combined.b64"
        ok, out, err = run_vm(cmd)
        if not ok:
            print(f"{label}: chunk write failed:", err or out)
            return False
    ok, out, err = run_vm(f"base64 -d /tmp/combined.b64 > {remote_path} && wc -c {remote_path}")
    if not ok:
        print(f"{label}: decode failed:", err or out)
        return False
    ok, out, err = run_vm(
        "python3 -c \"import hashlib; d=open('%s','rb').read(); print('REMOTE_SHA256='+hashlib.sha256(d).hexdigest())\"" % remote_path,
        timeout_sec=60,
    )
    if not ok:
        print(f"{label}: SHA check failed (continuing)")
    else:
        remote_sha = None
        for line in (out or "").splitlines():
            if "REMOTE_SHA256=" in line:
                remote_sha = line.split("REMOTE_SHA256=", 1)[1].strip().split()[0]
                break
        if remote_sha and remote_sha != local_sha:
            print(f"{label}: SHA256 mismatch (continuing anyway)")
    print(f"{label}: transferred")
    return True


def make_apply_server_patch_script(remote_ollama: str) -> bytes:
    """Build a Python script that patches server.go in place on the VM (needle->insert)."""
    needle3 = (
        "\tfor k, done := range extraEnvsDone {\n\t\tif !done {\n\t\t\tcmd.Env = append(cmd.Env, k+\"=\"+extraEnvs[k])\n\t\t}\n\t}\n\n"
        "\tslog.Info(\"starting runner\""
    )
    insert3 = (
        "\tfor k, done := range extraEnvsDone {\n\t\tif !done {\n\t\t\tcmd.Env = append(cmd.Env, k+\"=\"+extraEnvs[k])\n\t\t}\n\t}\n\n"
        "\t// Ensure runner gets LD_LIBRARY_PATH, OLLAMA_* (do NOT pass LD_PRELOAD: need real dlopen to load libggml-cuda.so)\n"
        "\tfor _, key := range []string{\"LD_LIBRARY_PATH\", \"OLLAMA_LIBRARY_PATH\", \"OLLAMA_LLM_LIBRARY\", \"OLLAMA_NUM_GPU\"} {\n"
        "\t\tif v, ok := os.LookupEnv(key); ok && v != \"\" {\n"
        "\t\t\tfound := false\n"
        "\t\t\tfor i := range cmd.Env {\n"
        "\t\t\t\tif strings.HasPrefix(cmd.Env[i], key+\"=\") {\n"
        "\t\t\t\t\tcmd.Env[i] = key + \"=\" + v\n"
        "\t\t\t\t\tfound = true\n"
        "\t\t\t\t\tbreak\n"
        "\t\t\t\t}\n"
        "\t\t\t}\n"
        "\t\t\tif !found {\n"
        "\t\t\t\tcmd.Env = append(cmd.Env, key+\"=\"+v)\n"
        "\t\t\t}\n"
        "\t\t}\n"
        "\t}\n"
        "\t// Remove LD_PRELOAD from runner so dlopen loads libggml-cuda.so\n"
        "\tn := 0\n"
        "\tfor _, e := range cmd.Env {\n"
        "\t\tif !strings.HasPrefix(e, \"LD_PRELOAD=\") {\n"
        "\t\t\tcmd.Env[n] = e\n"
        "\t\t\tn++\n"
        "\t\t}\n"
        "\t}\n"
        "\tcmd.Env = cmd.Env[:n]\n\n"
        "\tslog.Info(\"starting runner\""
    )
    script = f'''#! /usr/bin/env python3
import sys
path = "{remote_ollama}/llm/server.go"
needle = {repr(needle3)}
insert = {repr(insert3)}
with open(path, "r") as f:
    c = f.read()
if insert in c:
    print("SERVER_ALREADY_PATCHED")
elif needle not in c:
    print("NEEDLE_NOT_FOUND")
    sys.exit(1)
else:
    c = c.replace(needle, insert, 1)
    with open(path, "w") as f:
        f.write(c)
    print("PATCHED_SERVER")
# discover/runner.go: GPU lib dir first (initial bootstrap + refresh path)
runner_path = "{remote_ollama}/discover/runner.go"
old_d = "\\t\\t\\tdirs = []string{{ml.LibOllamaPath, dir}}"
new_d = "\\t\\t\\tdirs = []string{{dir, ml.LibOllamaPath}}"
old_refresh = "updatedDevices := bootstrapDevices(ctx, []string{{ml.LibOllamaPath, dir}}, devFilter)"
new_refresh = "updatedDevices := bootstrapDevices(ctx, []string{{dir, ml.LibOllamaPath}}, devFilter)"
with open(runner_path, "r") as f:
    r = f.read()
modified = False
if new_d not in r and old_d in r:
    r = r.replace(old_d, new_d, 1)
    modified = True
    print("PATCHED_DISCOVER")
elif new_d in r:
    pass  # already patched
else:
    print("DISCOVER_NEEDLE_NOT_FOUND")
if new_refresh not in r and old_refresh in r:
    r = r.replace(old_refresh, new_refresh, 1)
    modified = True
    print("PATCHED_DISCOVER_REFRESH")
if modified:
    with open(runner_path, "w") as f:
        f.write(r)
'''
    return script.encode("utf-8")


def fetch_from_vm(remote_path: str) -> str:
    """Get file content from VM (cat path). Strips connect_vm wrapper; keeps only Go source."""
    ok, out, _ = run_vm(f"cat {remote_path}", timeout_sec=30)
    if not ok:
        return ""
    raw = (out or "").splitlines()
    # Drop connect_vm trailer and any leading "Output:" / command echo
    start = 0
    for i, line in enumerate(raw):
        if "__CONNECT_VM_DONE__" in line:
            raw = raw[:i]
            break
    for i, line in enumerate(raw):
        if line.strip().startswith("package "):
            start = i
            break
    return "\n".join(raw[start:]) + "\n" if raw[start:] else ""


def main():
    from_vm = "--from-vm" in sys.argv
    if from_vm:
        sys.argv = [a for a in sys.argv if a != "--from-vm"]
        print("Applying server.go + discover/runner.go patch on VM...")
        script_bytes = make_apply_server_patch_script(REMOTE_OLLAMA)
        if not transfer_file_to_vm(script_bytes, "/tmp/apply_ollama_patch.py", "apply_patch.py"):
            return 1
        ok, out, err = run_vm(
            f"cd {REMOTE_OLLAMA} && python3 /tmp/apply_ollama_patch.py",
            timeout_sec=30,
        )
        print(out or err)
        if not ok or "NEEDLE_NOT_FOUND" in (out or ""):
            print("Patch script failed or server.go target not found.")
            return 1
        if "PATCHED_SERVER" not in (out or "") and "SERVER_ALREADY_PATCHED" not in (out or ""):
            print("Patch script did not confirm server state.")
            return 1
        print("Patched server.go (and discover/runner.go if needed). Building...")
        # Skip to build/install (no device/server/discover content to transfer)
        copy_build = (
            f"cd {REMOTE_OLLAMA} && ( /usr/local/go/bin/go version 2>/dev/null && /usr/local/go/bin/go build -o ollama.bin . || go build -o ollama.bin . ) 2>&1; echo BUILD_EXIT=$?"
        )
        ok, out, err = run_vm(copy_build, timeout_sec=300)
        print(out)
        if not ok:
            print("Build failed:", err or out)
            return 1
        if "BUILD_EXIT=0" not in (out or ""):
            print("Build did not succeed. Output above.")
            return 1
        install_cmd = (
            f"echo {VM_PASSWORD} | sudo -S systemctl stop ollama && "
            f"echo {VM_PASSWORD} | sudo -S cp {REMOTE_OLLAMA}/ollama.bin /usr/local/bin/ollama.bin && "
            f"echo {VM_PASSWORD} | sudo -S cp {REMOTE_OLLAMA}/ollama.bin /usr/local/bin/ollama.bin.real && "
            f"echo {VM_PASSWORD} | sudo -S systemctl start ollama"
        )
        ok, out, err = run_vm(install_cmd)
        print(out or err)
        if not ok:
            print("Install or restart failed")
            return 1
        print("Installed ollama.bin and restarted ollama service. Check discovery: journalctl -u ollama -n 25 | grep -E 'inference compute|initial_count'")
        return 0
    else:
        device_go = os.path.join(OLLAMA_SRC, "ml", "device.go")
        server_go = os.path.join(OLLAMA_SRC, "llm", "server.go")
        discover_runner_go = os.path.join(OLLAMA_SRC, "discover", "runner.go")
        if not os.path.isfile(device_go) or not os.path.isfile(server_go):
            print("Error: ollama-src/ml/device.go or llm/server.go not found (use --from-vm to patch VM files in place)")
            return 1
        if not os.path.isfile(discover_runner_go):
            print("Error: ollama-src/discover/runner.go not found")
            return 1
        with open(device_go, "r") as f:
            device_content = f.read()
        with open(server_go, "r") as f:
            server_content = f.read()
        with open(discover_runner_go, "r") as f:
            discover_content = f.read()

    device_content = patch_device_go(device_content, require=not from_vm)
    server_content = patch_server_go(server_content)
    discover_content = patch_discover_runner_go(discover_content)
    print("Patches applied (device.go, server.go, discover/runner.go: GPU lib dir first).")

    if not transfer_file_to_vm(device_content.encode("utf-8"), "/tmp/device_patched.go", "device.go"):
        return 1
    if not transfer_file_to_vm(server_content.encode("utf-8"), "/tmp/server_patched.go", "server.go"):
        return 1
    if not transfer_file_to_vm(discover_content.encode("utf-8"), "/tmp/runner_patched.go", "discover/runner.go"):
        return 1

    # Copy to ollama tree and build (use Go 1.26 if available; VM may have 1.18 in PATH)
    copy_build = (
        f"cp /tmp/device_patched.go {REMOTE_OLLAMA}/ml/device.go && "
        f"cp /tmp/server_patched.go {REMOTE_OLLAMA}/llm/server.go && "
        f"cp /tmp/runner_patched.go {REMOTE_OLLAMA}/discover/runner.go && "
        f"cd {REMOTE_OLLAMA} && ( /usr/local/go/bin/go version 2>/dev/null && /usr/local/go/bin/go build -o ollama.bin . || go build -o ollama.bin . ) 2>&1; echo BUILD_EXIT=$?"
    )
    ok, out, err = run_vm(copy_build, timeout_sec=300)
    print(out)
    if not ok:
        print("Copy or build failed:", err or out)
        return 1
    if "BUILD_EXIT=0" not in (out or ""):
        print("Build did not succeed. Output above.")
        return 1

    install_cmd = (
        f"echo {VM_PASSWORD} | sudo -S systemctl stop ollama && "
        f"echo {VM_PASSWORD} | sudo -S cp {REMOTE_OLLAMA}/ollama.bin /usr/local/bin/ollama.bin && "
        f"echo {VM_PASSWORD} | sudo -S cp {REMOTE_OLLAMA}/ollama.bin /usr/local/bin/ollama.bin.real && "
        f"echo {VM_PASSWORD} | sudo -S systemctl start ollama"
    )
    ok, out, err = run_vm(install_cmd)
    print(out or err)
    if not ok:
        print("Install or restart failed")
        return 1
    print("Installed ollama.bin and restarted ollama service.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
