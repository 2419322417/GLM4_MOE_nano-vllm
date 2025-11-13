#!/bin/bash
set -e

python3 -m nanovllm.models.glm4_moe.attention

python3 nanovllm/tests/attention.py
python3 nanovllm/example/attention.py
