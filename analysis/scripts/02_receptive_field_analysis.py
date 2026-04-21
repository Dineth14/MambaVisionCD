#!/usr/bin/env python3
"""
Gradient-based effective receptive field analysis for MambaVision-T-1K.

For a selected stage tensor y_l, the effective receptive field is estimated by
backpropagating a center spatial neuron to the input image and summing
|d y_l / d x| over RGB channels. The theoretical receptive field is propagated
through convolutional kernels with RF_l = RF_{l-1} + (k_l - 1) * jump_{l-1}.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HOME", "/tmp/mambavision_hf_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mambavision_mpl_cache")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


plt.style.use("dark_background")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "analysis" / "results" / "figures"
JSON_DIR = ROOT / "analysis" / "results" / "json"
REPORT_DIR = ROOT / "analysis" / "results" / "reports"
for directory in (FIG_DIR, JSON_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_model() -> torch.nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)


def call_model(model: torch.nn.Module, x: torch.Tensor) -> Any:
    try:
        return model(x)
    except TypeError:
        return model(pixel_values=x)


def find_stage_modules(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    for name, module in model.named_modules():
        if name.endswith("levels") and hasattr(module, "__len__"):
            try:
                return [(f"{name}.{idx}", module[idx]) for idx in range(min(4, len(module)))]
            except Exception:
                pass
    if hasattr(model, "levels"):
        return [(f"levels.{idx}", model.levels[idx]) for idx in range(min(4, len(model.levels)))]
    return []


def tensor_to_bchw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert common feature layouts to BCHW for center-spatial ERF logic."""
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 3:
        b, n, c = tensor.shape
        side = int(math.sqrt(n))
        if side * side == n:
            return tensor.transpose(1, 2).reshape(b, c, side, side)
    raise ValueError(f"Unsupported feature shape for ERF: {tuple(tensor.shape)}")


def stage_outputs_with_hooks(model: torch.nn.Module, x: torch.Tensor) -> Dict[int, torch.Tensor]:
    """Hook each stage and retain its tensor output for gradient extraction."""
    outputs: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(stage_idx: int):
        def _hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
            candidate = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(candidate):
                candidate.retain_grad()
                outputs[stage_idx] = candidate

        return _hook

    stages = find_stage_modules(model)
    if not stages:
        raise RuntimeError("Could not locate four stage modules. Check the HF remote-code model structure.")
    for idx, (_name, module) in enumerate(stages[:4], start=1):
        handles.append(module.register_forward_hook(make_hook(idx)))

    call_model(model, x)
    for handle in handles:
        handle.remove()
    return outputs


def compute_erf_for_stage(model: torch.nn.Module, device: torch.device, stage_idx: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
    model.zero_grad(set_to_none=True)
    x = torch.ones(1, 3, 224, 224, device=device, requires_grad=True)
    outputs = stage_outputs_with_hooks(model, x)
    feature = tensor_to_bchw(outputs[stage_idx])
    _, _, h, w = feature.shape

    # A scalar center response is enough: its gradient indicates which input
    # pixels can influence that stage's center spatial location.
    target = feature[0, :, h // 2, w // 2].mean()
    target.backward()
    grad = x.grad.detach().abs().sum(dim=1)[0]
    grad = grad / (grad.max() + 1e-12)
    return grad.cpu().numpy(), tuple(feature.shape)


def energy_radii(heatmap: np.ndarray) -> Dict[str, float]:
    yy, xx = np.indices(heatmap.shape)
    cy, cx = (np.array(heatmap.shape) - 1) / 2.0
    radii = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).reshape(-1)
    weights = np.maximum(heatmap.reshape(-1), 0)
    order = np.argsort(radii)
    radii_sorted = radii[order]
    cumulative = np.cumsum(weights[order])
    total = cumulative[-1] + 1e-12
    return {
        f"r{int(frac * 100)}": float(radii_sorted[np.searchsorted(cumulative, frac * total)])
        for frac in (0.50, 0.75, 0.90)
    }


def effective_area(heatmap: np.ndarray, threshold: float = 0.05) -> int:
    return int((heatmap >= threshold * heatmap.max()).sum())


def theoretical_rf_by_stage(model: torch.nn.Module) -> Dict[int, Dict[str, Any]]:
    """Approximate RF from encountered Conv2d layers and assign values to stages by name."""
    rf = 1
    jump = 1
    stage_rf: Dict[int, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
        rf = rf + (kernel - 1) * jump
        jump *= stride
        stage = None
        if "levels.0" in name:
            stage = 1
        elif "levels.1" in name:
            stage = 2
        elif "levels.2" in name:
            stage = 3
        elif "levels.3" in name:
            stage = 4
        elif "patch_embed" in name:
            stage = 0
        if stage:
            stage_rf[stage] = {"theoretical_rf": rf, "effective_stride": jump, "last_conv": name}
    return stage_rf


def plot_erf_heatmaps(erfs: Dict[int, np.ndarray]) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for idx, ax in enumerate(axes, start=1):
        heat = erfs[idx]
        log_heat = np.log10(heat + 1e-8)
        im = ax.imshow(log_heat, cmap="magma")
        ax.set_title(f"Stage {idx}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Effective Receptive Field per Stage, log10(|gradient|)", fontsize=14)
    out_path = FIG_DIR / "erf_per_stage.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_contour_overlay(erfs: Dict[int, np.ndarray], radii: Dict[int, Dict[str, float]]) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    base = np.full((224, 224), 0.42)
    for idx, ax in enumerate(axes, start=1):
        heat = erfs[idx]
        ax.imshow(base, cmap="gray", vmin=0, vmax=1)
        ax.contour(heat, levels=[0.10, 0.25, 0.50], colors=["#00d4ff", "#39ff8f", "#ffcc66"], linewidths=1.1)
        cy = cx = 111.5
        for label, color in (("r50", "#00d4ff"), ("r75", "#39ff8f"), ("r90", "#ffcc66")):
            circle = plt.Circle((cx, cy), radii[idx][label], edgecolor=color, facecolor="none", lw=1.2, ls="--")
            ax.add_patch(circle)
        ax.set_title(f"Stage {idx}: 50/75/90% energy")
        ax.set_xticks([])
        ax.set_yticks([])
    out_path = FIG_DIR / "erf_contour_overlay.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compare_ssm_attention_erf(model: torch.nn.Module, device: torch.device) -> List[Dict[str, Any]]:
    """Probe selected SSM and attention blocks in stages 3/4 and compare gradient reach."""
    selected = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if ("levels.2" in name or "levels.3" in name) and cls in {"MambaVisionMixer", "Attention"}:
            selected.append((name, module, cls))
    results = []
    for name, module, cls in selected[:8]:
        captured: Dict[str, torch.Tensor] = {}

        def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
            if torch.is_tensor(output):
                output.retain_grad()
                captured["out"] = output

        handle = module.register_forward_hook(hook)
        model.zero_grad(set_to_none=True)
        x = torch.ones(1, 3, 224, 224, device=device, requires_grad=True)
        call_model(model, x)
        handle.remove()
        out = captured.get("out")
        if out is None:
            continue
        if out.ndim == 3:
            target = out[0, out.shape[1] // 2, :].mean()
        elif out.ndim == 4:
            target = out[0, :, out.shape[2] // 2, out.shape[3] // 2].mean()
        else:
            continue
        target.backward()
        heat = x.grad.detach().abs().sum(dim=1)[0].cpu().numpy()
        heat = heat / (heat.max() + 1e-12)
        radii = energy_radii(heat)
        results.append(
            {
                "name": name,
                "type": "SSM" if cls == "MambaVisionMixer" else "Attention",
                "r90_pixels": radii["r90"],
                "active_area_threshold_5pct": effective_area(heat, 0.05),
                "interpretation": "global-leaning" if radii["r90"] > 70 else "local-to-midrange",
            }
        )
    return results


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading nvidia/MambaVision-T-1K on {device}...")
    model = load_model().to(device).eval()

    erfs: Dict[int, np.ndarray] = {}
    feature_shapes: Dict[int, Tuple[int, ...]] = {}
    for stage_idx in range(1, 5):
        print(f"Computing ERF for stage {stage_idx}...")
        erfs[stage_idx], feature_shapes[stage_idx] = compute_erf_for_stage(model, device, stage_idx)

    radii = {idx: energy_radii(heat) for idx, heat in erfs.items()}
    theory = theoretical_rf_by_stage(model)
    ssm_attention = compare_ssm_attention_erf(model, device)

    heatmap_path = plot_erf_heatmaps(erfs)
    overlay_path = plot_contour_overlay(erfs, radii)

    rows = []
    print("\nTheoretical vs effective receptive field")
    print("=" * 86)
    print(f"{'stage':>5s} {'feature shape':>22s} {'theory RF':>10s} {'r50':>8s} {'r75':>8s} {'r90':>8s} {'area@5%':>10s}")
    for idx in range(1, 5):
        row = {
            "stage": idx,
            "feature_shape": feature_shapes[idx],
            "theoretical": theory.get(idx, {}),
            "effective_energy_radii": radii[idx],
            "active_area_threshold_5pct": effective_area(erfs[idx], 0.05),
        }
        rows.append(row)
        print(
            f"{idx:5d} {str(feature_shapes[idx]):>22s} "
            f"{theory.get(idx, {}).get('theoretical_rf', 'n/a'):>10} "
            f"{radii[idx]['r50']:8.1f} {radii[idx]['r75']:8.1f} {radii[idx]['r90']:8.1f} "
            f"{row['active_area_threshold_5pct']:10d}"
        )

    print("\nStage 3/4 SSM vs attention ERF comparison")
    print("=" * 86)
    for item in ssm_attention:
        print(f"{item['type']:9s} {item['name'][:48]:48s} r90={item['r90_pixels']:.1f}px {item['interpretation']}")

    json_path = JSON_DIR / "receptive_field_analysis.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": "nvidia/MambaVision-T-1K",
                "stage_comparison": rows,
                "ssm_vs_attention": ssm_attention,
                "figures": [str(heatmap_path.relative_to(ROOT)), str(overlay_path.relative_to(ROOT))],
            },
            f,
            indent=2,
        )

    (REPORT_DIR / "02_receptive_field_analysis.md").write_text(
        "# Receptive Field Analysis\n\n"
        f"- ERF heatmaps: `{heatmap_path.relative_to(ROOT)}`\n"
        f"- ERF overlay: `{overlay_path.relative_to(ROOT)}`\n"
        f"- Structured results: `{json_path.relative_to(ROOT)}`\n",
        encoding="utf-8",
    )
    print(f"\nSaved {heatmap_path}, {overlay_path}, and {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
