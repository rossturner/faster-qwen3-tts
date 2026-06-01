# Two-model coexistence spike result (2026-06-01)

**Spike:** `spikes/two_model_coexistence.py` — load CustomVoice (1.7B) and Base (1.7B) together on one RTX 4090 (24 GB), generate one clip from each.

## Outcome: PASS

Both models loaded simultaneously, each captured its own CUDA graphs, and both generated successfully on the single GPU.

| Metric | Value |
| --- | --- |
| Peak VRAM allocated | **9.07 GB** (well under the 24 GB available) |
| CustomVoice clip (`aiden`, English) | 1.84 s |
| Base clone clip (English, `en_f_v1_ref.wav`) | 1.84 s |

Verify command: `.venv/bin/python spikes/two_model_coexistence.py` — printed both durations and the peak VRAM line, exited 0.

## Decision

Proceed with both-models design.
