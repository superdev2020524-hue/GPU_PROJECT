#!/usr/bin/env python3
"""Patch llm/server.go: instrument waitUntilRunnerLaunched (Phase3). Run on VM."""
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "/home/test-4/ollama/llm/server.go"
with open(path) as f:
    c = f.read()

if "Phase3 waitUntilRunnerLaunched" in c:
    print("ALREADY")
    sys.exit(0)

old = """// waitUntilRunnerLaunched sleeps until the runner subprocess is alive enough
// to respond to status requests
func (s *llmServer) waitUntilRunnerLaunched(ctx context.Context) error {
	for {
		_, err := s.getServerStatus(ctx)
		if err == nil {
			break
		}

		t := time.NewTimer(10 * time.Millisecond)
		select {
		case <-t.C:
			continue
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return nil
}"""

new = """// waitUntilRunnerLaunched sleeps until the runner subprocess is alive enough
// to respond to status requests
func (s *llmServer) waitUntilRunnerLaunched(ctx context.Context) error {
	var pollN int
	for {
		pollN++
		_, err := s.getServerStatus(ctx)
		if err == nil {
			slog.Info("Phase3 waitUntilRunnerLaunched runner responded", "port", s.port, "polls", pollN)
			break
		}
		if pollN == 1 {
			slog.Info("Phase3 waitUntilRunnerLaunched waiting for runner", "port", s.port)
		} else if pollN%50 == 0 {
			slog.Info("Phase3 waitUntilRunnerLaunched poll", "port", s.port, "polls", pollN, "err", err)
		}
		t := time.NewTimer(10 * time.Millisecond)
		select {
		case <-t.C:
			continue
		case <-ctx.Done():
			slog.Info("Phase3 waitUntilRunnerLaunched timeout", "port", s.port, "polls", pollN)
			return ctx.Err()
		}
	}

	return nil
}"""

if old not in c:
    print("NOT_FOUND")
    sys.exit(1)
with open(path, "w") as f:
    f.write(c.replace(old, new, 1))
print("PATCHED")
