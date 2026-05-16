# Source with: gdb -q -x gdb_sigfpe_capture.gdb --args /usr/local/bin/ollama.bin serve
# Or batch: gdb -q -batch -x gdb_sigfpe_capture.gdb --args /usr/local/bin/ollama.bin serve
# See CRASH_SYMBOLICATION_AND_COREDUMPS.md §4.

set pagination off
set confirm off
# Go binaries often exit immediately under GDB if the default /bin/sh -c wrapper is used.
set startup-with-shell off
set follow-fork-mode child
set detach-on-fork off

catch signal SIGFPE
commands
  silent
  printf "\n========== GDB: SIGFPE ==========\n"
  thread apply all bt full
  quit
end

run
