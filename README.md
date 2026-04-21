# MambaVision Deep Analysis & Visualization Repository

This fork adds a structured analysis and documentation layer on top of NVIDIA's MambaVision codebase. It keeps the original model implementation intact and adds:

- `analysis/scripts/`: runnable probes for architecture, receptive field, frequency response, edge sensitivity, activations, SSM dynamics, and mathematical verification.
- `analysis/results/`: generated figures, JSON outputs, and markdown reports.
- `docs/math_notes.md`: LaTeX-annotated mathematical notes for MambaVision and change detection use.
- `docs/website/`: a Vite/Node interactive research portal with copied result JSON/figures for GitHub Pages deployment.

The scripts load the HuggingFace model:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
```

For this machine, the verified environment is:

```bash
/userhomes/keshawa17/anaconda3/envs/mamba_new/bin/python
```

`mamba_ssm` requires CUDA tensors for these HuggingFace models. Run the analysis from a CUDA-visible shell.

## Environment Setup

Use the existing local environment if it already runs MambaVision. For a fresh environment:

```bash
conda create -n mambavision-analysis python=3.10 -y
conda activate mambavision-analysis
pip install -r requirements.txt
pip install transformers matplotlib numpy pillow einops timm mamba-ssm
```

Install a PyTorch build that matches your CUDA/CPU environment before running GPU probes. All analysis scripts choose:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Run the Analysis

Run scripts from the repository root:

```bash
python analysis/scripts/01_model_architecture_map.py
python analysis/scripts/02_receptive_field_analysis.py
python analysis/scripts/03_frequency_phase_response.py
python analysis/scripts/04_edge_detection_probing.py
python analysis/scripts/05_layer_activation_visualizer.py
python analysis/scripts/06_ssm_eigenspectrum.py
python analysis/scripts/07_mathematical_verification.py
python analysis/scripts/08_model_family_analysis.py --models tiny tiny2 small base large
```

Recommended order:

1. `01_model_architecture_map.py`: smoke test, module tree, architecture JSON, stage diagram.
2. `02_receptive_field_analysis.py`: gradient effective receptive field and RF overlays.
3. `03_frequency_phase_response.py`: sinusoidal grating response, FFT power, phase response.
4. `04_edge_detection_probing.py`: edge stimuli, stem filters, edge selectivity index.
5. `05_layer_activation_visualizer.py`: activation grids, branch distributions, attention maps.
6. `06_ssm_eigenspectrum.py`: SSM eigenvalues, impulse response, frequency response, delta distribution.
7. `07_mathematical_verification.py`: selective scan, branch fusion, resolution reduction, throughput timing.
8. `08_model_family_analysis.py`: compact cross-model suite for MambaVision-T, T2, S, B, and L with consolidated reasoning report.

Only script 01 is intended as the light smoke test. Scripts 02-08 require a CUDA-visible environment here because `mamba_ssm` selective scan is CUDA-backed.

## Expected Outputs

Generated files are written under:

```text
analysis/results/
├── figures/   # PNG visualizations
├── json/      # pretty-printed structured results
└── reports/   # auto-generated markdown summaries
```

Important figures include:

- `mambavision_stage_diagram.png`
- `erf_per_stage.png`
- `frequency_tuning_per_stage.png`
- `stem_filters.png`
- `layer_function_summary.png`
- `ssm_eigenspectrum.png`
- `family_erf_r90.png`
- `family_throughput.png`
- `family_edge_top_esi.png`
- `family_frequency_dominance.png`
- `family_ssm_summary.png`

The consolidated model-family result is:

```text
analysis/results/json/model_family_analysis.json
analysis/results/reports/08_model_family_analysis.md
```

## Website

The website is now a deployable Vite project:

```bash
cd docs/website
npm install
npm run dev
```

Build static assets for deployment:

```bash
npm run build
npm run preview
```

The app reads:

```text
docs/website/public/data/model_family_analysis.json
docs/website/public/results/figures/
```

To refresh the website after rerunning analysis:

```bash
cp analysis/results/json/model_family_analysis.json docs/website/public/data/model_family_analysis.json
cp analysis/results/figures/*.png docs/website/public/results/figures/
```

A GitHub Pages workflow is included at:

```text
.github/workflows/deploy-website.yml
```

Note: this shell did not have `node`/`npm` installed, so the website source and data references were statically verified here, but the production Vite build should be run in an environment with Node 20+.

## Mathematical Notes

See:

```text
docs/math_notes.md
```

The notes cover:

- MambaVision Mixer equations.
- Continuous-to-discrete SSM derivation.
- Selective scan parameterization.
- Symmetric non-SSM branch.
- Relationship to VMamba, Vim, and Swin.
- Change detection usage as a Siamese encoder.

## Original MambaVision

Original project:

- GitHub: <https://github.com/NVlabs/MambaVision>
- Paper: <https://arxiv.org/abs/2407.08083>
- HuggingFace model: <https://huggingface.co/nvidia/MambaVision-T-1K>

Citation:

```bibtex
@article{hatamizadeh2024mambavision,
  title={MambaVision: A Hybrid Mamba-Transformer Vision Backbone},
  author={Hatamizadeh, Ali and Kautz, Jan},
  journal={arXiv preprint arXiv:2407.08083},
  year={2024}
}
```

## Attribution

Original MambaVision implementation and pretrained models are by NVIDIA Research. This repository layer adds local analysis scripts, generated reports, mathematical documentation, and the interactive visualization website for research and change detection exploration.
