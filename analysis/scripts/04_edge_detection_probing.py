#!/usr/bin/env python3
"""
Probe edge selectivity in MambaVision.

The script builds controlled edge stimuli, records convolution/SSM activations,
computes activation magnitude, activation spatial gradients, Sobel preservation,
and ranks layers by an edge selectivity index:

    ESI = (response_edge - response_flat) / (response_edge + response_flat)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, device=device) - (size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size)


def blur_image(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    size = max(3, int(2 * round(3 * sigma) + 1))
    kernel = gaussian_kernel(size, sigma, x.device).repeat(3, 1, 1, 1)
    return F.conv2d(x.unsqueeze(0), kernel, padding=size // 2, groups=3).squeeze(0)


def edge_stimuli(device: torch.device, size: int = 224) -> Dict[str, torch.Tensor]:
    y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
    center = (size - 1) / 2
    stimuli: Dict[str, torch.Tensor] = {"flat": torch.full((3, size, size), 0.5, device=device)}
    masks = {
        "horizontal_step": y > center,
        "vertical_step": x > center,
        "diagonal_45_step": (x + y) > size,
        "diagonal_135_step": (x - y) > 0,
    }
    for name, mask in masks.items():
        image = mask.float()
        stimuli[name] = image.repeat(3, 1, 1)
    for sigma in (1, 3, 5):
        stimuli[f"vertical_blur_sigma_{sigma}"] = blur_image(stimuli["vertical_step"], float(sigma))
    ramp_x = torch.clamp((x.float() - center + 32) / 64, 0, 1)
    ramp_y = torch.clamp((y.float() - center + 32) / 64, 0, 1)
    stimuli["vertical_ramp"] = ramp_x.repeat(3, 1, 1)
    stimuli["horizontal_ramp"] = ramp_y.repeat(3, 1, 1)
    stimuli["diagonal_ramp"] = torch.clamp((x.float() + y.float() - size + 45) / 90, 0, 1).repeat(3, 1, 1)
    stimuli["thin_vertical_line"] = ((x - center).abs() < 2).float().repeat(3, 1, 1)
    stimuli["checker_edge_texture"] = (((x // 16 + y // 16) % 2).float()).repeat(3, 1, 1)
    return stimuli


def synthetic_aerial(device: torch.device, size: int = 224) -> torch.Tensor:
    y, x = torch.meshgrid(torch.linspace(0, 1, size, device=device), torch.linspace(0, 1, size, device=device), indexing="ij")
    base = 0.35 + 0.25 * torch.sin(18 * x) * torch.cos(14 * y)
    roads = ((torch.abs(y - 0.35 - 0.08 * torch.sin(8 * x)) < 0.015) | (torch.abs(x - 0.68) < 0.012)).float()
    fields = (((x * 7).floor() + (y * 5).floor()) % 2) * 0.12
    image = torch.stack([base + fields, base + 0.1 * roads, base + 0.3 * roads], dim=0).clamp(0, 1)
    return image


def spatial_gradient_magnitude(feature: torch.Tensor) -> torch.Tensor:
    fmap = feature.detach().float()
    if fmap.ndim == 3:
        fmap = fmap.unsqueeze(0)
    if fmap.ndim == 4:
        fmap = fmap.mean(dim=1, keepdim=True)
    dx = fmap[..., :, 1:] - fmap[..., :, :-1]
    dy = fmap[..., 1:, :] - fmap[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx ** 2 + dy ** 2 + 1e-12)


def sobel_response(feature: torch.Tensor) -> float:
    fmap = feature.detach().float()
    if fmap.ndim == 3:
        fmap = fmap.unsqueeze(0)
    if fmap.ndim == 4:
        fmap = fmap.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=fmap.dtype, device=fmap.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(fmap, kx, padding=1)
    gy = F.conv2d(fmap, ky, padding=1)
    return float(torch.sqrt(gx ** 2 + gy ** 2 + 1e-12).mean().cpu())


def modules_to_hook(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    selected = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        in_early_stages = any(token in name for token in ("levels.0", "levels.1", "layer.0", "layer.1", "stage1", "stage2"))
        if isinstance(module, torch.nn.Conv2d) or (in_early_stages and cls == "MambaVisionMixer"):
            selected.append((name, module))
    return selected


def collect_activations(model: torch.nn.Module, x: torch.Tensor, selected: List[Tuple[str, torch.nn.Module]]) -> Dict[str, torch.Tensor]:
    activations: Dict[str, torch.Tensor] = {}
    handles = []
    for name, module in selected:
        def make_hook(layer_name: str):
            def _hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                candidate = output[0] if isinstance(output, (tuple, list)) else output
                if torch.is_tensor(candidate):
                    activations[layer_name] = candidate.detach()
            return _hook
        handles.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        call_model(model, x)
    for handle in handles:
        handle.remove()
    return activations


def first_conv(model: torch.nn.Module) -> Tuple[str, torch.nn.Conv2d]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            return name, module
    raise RuntimeError("No Conv2d layer found.")


def plot_stem_filters(model: torch.nn.Module) -> Path:
    name, conv = first_conv(model)
    weights = conv.weight.detach().float().cpu()
    n = min(weights.shape[0], 64)
    cols = 8
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx < n:
            kernel = weights[idx]
            if kernel.shape[0] >= 3:
                img = kernel[:3].permute(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min() + 1e-12)
                ax.imshow(img)
            else:
                ax.imshow(kernel[0], cmap="coolwarm")
    fig.suptitle(f"First convolution filters: {name}")
    out = FIG_DIR / "stem_filters.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_stem_filter_responses(model: torch.nn.Module, stimuli: Dict[str, torch.Tensor]) -> Path:
    name, conv = first_conv(model)
    device = next(model.parameters()).device
    names = [key for key in stimuli.keys() if key != "flat"][:12]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    axes = axes.reshape(-1)
    with torch.no_grad():
        for ax, stim_name in zip(axes, names):
            response = conv(stimuli[stim_name].unsqueeze(0).to(device)).abs().mean(dim=1)[0].cpu().numpy()
            ax.imshow(response, cmap="magma")
            ax.set_title(stim_name.replace("_", " "), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Mean absolute response of first conv: {name}")
    out = FIG_DIR / "stem_filter_responses.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def stage_features(model: torch.nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    for name, module in model.named_modules():
        if name.endswith("levels") and hasattr(module, "__len__"):
            for idx in range(min(4, len(module))):
                def make_hook(stage_idx: int):
                    def _hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                        candidate = output[0] if isinstance(output, (tuple, list)) else output
                        if torch.is_tensor(candidate):
                            captured[stage_idx] = candidate.detach()
                    return _hook
                handles.append(module[idx].register_forward_hook(make_hook(idx)))
            break
    with torch.no_grad():
        result = call_model(model, x)
    for handle in handles:
        handle.remove()
    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, (tuple, list)) and len(item) >= 4 and all(torch.is_tensor(t) for t in item[:4]):
                return [t.detach() for t in item[:4]]
    if len(captured) >= 4:
        return [captured[idx] for idx in range(4)]
    raise RuntimeError("Could not extract stage features.")


def plot_stage_maps(model: torch.nn.Module, device: torch.device) -> List[Path]:
    image = synthetic_aerial(device).unsqueeze(0)
    perturbed = (image + 0.06 * torch.randn_like(image)).clamp(0, 1)
    features_1 = stage_features(model, image)
    features_2 = stage_features(model, perturbed)
    paths = []
    for idx, (f1, f2) in enumerate(zip(features_1, features_2), start=1):
        act = f1.detach().abs().mean(dim=1)[0].cpu().numpy()
        diff = (f2 - f1).detach().abs().mean(dim=1)[0].cpu().numpy()
        for kind, data, cmap in (("activation_map", act, "viridis"), ("change_sensitivity", diff, "inferno")):
            fig, ax = plt.subplots(figsize=(5.5, 5))
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(f"{kind.replace('_', ' ').title()} Stage {idx}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            out = FIG_DIR / f"{kind}_stage{idx}.png"
            fig.savefig(out, dpi=220, bbox_inches="tight")
            plt.close(fig)
            paths.append(out)
    return paths


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading nvidia/MambaVision-T-1K on {device}...")
    model = load_model().to(device).eval()
    stimuli = edge_stimuli(device)
    selected = modules_to_hook(model)
    print(f"Hooking {len(selected)} convolution/early SSM layers.")

    flat_activations = collect_activations(model, stimuli["flat"].unsqueeze(0), selected)
    layer_records: Dict[str, Dict[str, Any]] = {
        name: {"layer": name, "class": module.__class__.__name__, "stimuli": {}} for name, module in selected
    }
    for stim_name, image in stimuli.items():
        activations = collect_activations(model, image.unsqueeze(0), selected)
        for layer_name, activation in activations.items():
            layer_records[layer_name]["stimuli"][stim_name] = {
                "mean_abs_activation": float(activation.abs().mean().cpu()),
                "spatial_gradient_magnitude": float(spatial_gradient_magnitude(activation).mean().cpu()),
                "sobel_response": sobel_response(activation),
            }

    ranked = []
    for layer_name, record in layer_records.items():
        flat_response = record["stimuli"].get("flat", {}).get("mean_abs_activation", 0.0)
        edge_values = [
            value["mean_abs_activation"]
            for key, value in record["stimuli"].items()
            if key != "flat" and "step" in key
        ]
        edge_response = float(np.mean(edge_values)) if edge_values else 0.0
        esi = (edge_response - flat_response) / (edge_response + flat_response + 1e-12)
        record["edge_response_mean"] = edge_response
        record["flat_response"] = flat_response
        record["edge_selectivity_index"] = float(esi)
        ranked.append(record)
    ranked.sort(key=lambda item: item["edge_selectivity_index"], reverse=True)

    stem_filter_path = plot_stem_filters(model)
    stem_response_path = plot_stem_filter_responses(model, stimuli)
    stage_paths = plot_stage_maps(model, device)

    print("\nRanked edge selectivity index")
    print("=" * 92)
    print(f"{'rank':>4s} {'ESI':>10s} {'edge':>12s} {'flat':>12s} layer")
    for rank, item in enumerate(ranked[:25], start=1):
        print(
            f"{rank:4d} {item['edge_selectivity_index']:10.4f} "
            f"{item['edge_response_mean']:12.5f} {item['flat_response']:12.5f} {item['layer']}"
        )

    json_path = JSON_DIR / "edge_detection_probing.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stimuli": list(stimuli.keys()),
                "ranked_layers": ranked,
                "figures": [str(stem_filter_path.relative_to(ROOT)), str(stem_response_path.relative_to(ROOT))]
                + [str(path.relative_to(ROOT)) for path in stage_paths],
            },
            f,
            indent=2,
        )
    (REPORT_DIR / "04_edge_detection_probing.md").write_text(
        "# Edge Detection Probing\n\n"
        f"- Stem filters: `{stem_filter_path.relative_to(ROOT)}`\n"
        f"- Stem responses: `{stem_response_path.relative_to(ROOT)}`\n"
        f"- JSON: `{json_path.relative_to(ROOT)}`\n",
        encoding="utf-8",
    )
    print(f"\nSaved edge probing JSON to {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
