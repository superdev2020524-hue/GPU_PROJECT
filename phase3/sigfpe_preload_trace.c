/*
 * Phase 3: preload hook to print a native backtrace on SIGFPE (E3).
 * Install only for short repro; remove LD_PRELOAD after capture.
 * Build on VM: gcc -shared -fPIC -O1 -o sigfpe_preload_trace.so sigfpe_preload_trace.c
 */
#define _GNU_SOURCE
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

static void sigfpe_handler(int sig, siginfo_t *info, void *ucontext)
{
	(void)sig;
	(void)info;
	(void)ucontext;
	void *buf[80];
	int n = backtrace(buf, 80);
	const char msg[] = "\n=== Phase3 SIGFPE (LD_PRELOAD) backtrace ===\n";
	(void)write(2, msg, sizeof(msg) - 1);
	backtrace_symbols_fd(buf, n, 2);
	(void)write(2, "=== end ===\n", 12);
	_exit(128 + SIGFPE);
}

static void init(void) __attribute__((constructor));
static void init(void)
{
	const char hi[] = "\n=== sigfpe_preload_trace.so loaded ===\n";
	(void)write(2, hi, sizeof(hi) - 1);
	struct sigaction sa;
	sa.sa_sigaction = sigfpe_handler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
	if (sigaction(SIGFPE, &sa, NULL) != 0)
		perror("sigaction SIGFPE");
}
