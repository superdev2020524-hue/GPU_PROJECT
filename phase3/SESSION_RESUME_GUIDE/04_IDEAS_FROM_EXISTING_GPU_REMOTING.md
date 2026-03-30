# Ideas from Existing GPU Remoting Work

This file is not a migration plan. It is a reminder that older CUDA remoting systems may still contain useful arguments and debugging ideas.

## 1. Why look at them

The project does not need to switch to CRICKET, rCUDA, or GVirtuS right now.

But they may still help answer questions like:

- which CUDA calls should be answered locally vs remotely
- how capability reporting can steer bad kernel selection
- how much API coverage is needed before applications become stable
- where async GPU faults really surface
- how to avoid application-specific hacks

## 2. Important repo clue already present

The repo is not unaware of this area.

`phase3/command.txt` explicitly says the `dlopen` / `dlsym` interception technique is standard in CUDA remoting systems such as:

- CRICKET
- rCUDA
- GVirtuS

So these should be treated as sources of design and debugging ideas, not as evidence that Phase 3 was built in ignorance.

## 3. What to mine from them

If reviewing those systems, look specifically for arguments about:

- driver API interception completeness
- module loading / fatbin handling
- context ownership and lifetime
- stream and event semantics
- capability and device-property reporting
- sync-point error surfacing
- when to emulate locally vs when to remote
- how remote GPU systems keep the guest experience "normal" for applications

## 4. How that helps the current blocker

For the current state, the most relevant ideas are:

- a stable remoting stack must avoid misleading capability or feature exposure
- async kernel faults often appear later at sync points, not at the original launch site
- correctness beats optimization when proving end-to-end viability
- forcing a simpler execution path is often a valid milestone strategy

Those are directly relevant to the current MMQ / graph-reserve problem.

## 5. Recommended use

Use those systems in a narrow way:

- read them for reasoning
- borrow diagnostic questions
- borrow architectural cautions
- do not derail the current Phase 1 push into a rewrite or platform switch

## 6. Working conclusion

Current best view:

- those older systems are useful as idea mines
- they do not currently appear to be drop-in replacements for the exact `VGPU-STUB -> mediator -> H100` design and constraints in this repo
- therefore they are supporting references, not the next action
