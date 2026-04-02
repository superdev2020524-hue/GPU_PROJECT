#!/usr/bin/env python3
import ctypes
import mmap
import os
import struct

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
PFN_MASK = (1 << 55) - 1


def effective_caps():
    with open("/proc/self/status", "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("CapEff:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def probe(size_mb):
    size = size_mb * 1024 * 1024
    mm = mmap.mmap(
        -1,
        size,
        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    mm[:] = b"\0" * size

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    rc = libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    err = ctypes.get_errno()

    base_vfn = addr // PAGE_SIZE
    total_pages = size // PAGE_SIZE
    entries = []
    with open("/proc/self/pagemap", "rb", buffering=0) as f:
        for i in range(total_pages):
            f.seek((base_vfn + i) * 8)
            entries.append(struct.unpack("Q", f.read(8))[0])

    present = sum(1 for e in entries if (e >> 63) & 1)
    pfns = [e & PFN_MASK for e in entries]
    nonzero_pfns = sum(1 for p in pfns if p != 0)

    best_run = 0
    cur_run = 0
    prev = None
    for pfn in pfns:
        if pfn == 0:
            cur_run = 0
        elif prev is not None and pfn == prev + 1:
            cur_run += 1
        else:
            cur_run = 1
        if cur_run > best_run:
            best_run = cur_run
        prev = pfn if pfn != 0 else None

    sample = []
    for idx, pfn in enumerate(pfns):
        if pfn != 0:
            sample.append(f"{idx}:0x{pfn:x}")
        if len(sample) >= 16:
            break

    print(
        "RAW_PAGEMAP"
        f" size_mb={size_mb}"
        f" mlock_rc={rc}"
        f" errno={err}"
        f" capeff=0x{effective_caps()}"
        f" present_pages={present}"
        f" nonzero_pfns={nonzero_pfns}"
        f" best_run_pages={best_run}"
        f" best_run_kb={(best_run * PAGE_SIZE) // 1024}"
    )
    print("RAW_PAGEMAP_SAMPLE " + (" ".join(sample) if sample else "none"))


def main():
    for size_mb in (4, 64, 256):
        probe(size_mb)


if __name__ == "__main__":
    main()
