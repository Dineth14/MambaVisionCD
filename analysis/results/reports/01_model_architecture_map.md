# MambaVision Architecture Map

- Model: `nvidia/MambaVision-T-1K`
- Total parameters: `31,794,248`
- JSON: `analysis/results/json/architecture_graph.json`
- Stage diagram: `analysis/results/figures/mambavision_stage_diagram.png`

## Stages

- Stage 1: Conv, C=80, R=unknown, depth=1
- Stage 2: Conv, C=160, R=unknown, depth=3
- Stage 3: Mamba+Attn, C=320, R=unknown, depth=8
- Stage 4: Mamba+Attn, C=640, R=unknown, depth=4