# Experiment 01: ASCII Box

Spec (from root README): "Každé slovo vstupu do ASCII boxu na výstup."

Goals:
- Probe if a base/instruct model can do it via few-shot prompting.
- Later: train a small adapter and test transfer to chat/instruct.

Run baseline:
- Prepare a Python 3.12 env and install requirements at repo root.
- Example:
  - MODEL: Qwen/Qwen2.5-0.5B-Instruct (changeable)
  - Inputs: provided via CLI

Outputs land under `outputs/` as JSONL.
