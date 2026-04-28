# 09 - TWA Research

## Purpose

Start the longer-term TWA and future compute API research track without
destabilizing the vGPU engineering baseline.

## Start Conditions

- Ollama baseline preserved;
- general CUDA gate passing;
- at least one non-Ollama framework gate passing or actively isolated;
- user approves research work as a parallel track.

## Research Questions

- which existing APIs or libraries can express TWA-style workloads;
- whether CUDA-like semantics are enough;
- whether a new operation set is required;
- what a minimal software prototype should prove before hardware work;
- how future hardware would expose itself to applications.

## Deliverables

- research memo;
- existing API comparison;
- minimal prototype proposal;
- risks and unknowns.
