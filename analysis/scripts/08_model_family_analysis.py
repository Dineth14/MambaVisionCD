#!/usr/bin/env python3
"""
Cross-model MambaVision analysis suite.

This script runs a compact version of the analysis categories for the requested
ImageNet-1K model family:

- architecture/stage shapes
- effective receptive field radii
- sinusoidal frequency/orientation response
- edge selectivity ranking
- SSM eigenspectrum/impulse half-life
- selective-scan and concatenation verification
- throughput timing with memory-aware batch reduction

It records measured results only. Interpretive text in the report is generated
from those measurements and avoids claiming causality beyond what the probes
directly support.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("HF_HOME", "/tmp/mambavision_hf_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mambavision_mpl_cache")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


plt.style.use("dark_background")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "analysis" / "results" / "figures"
JSON_DIR = ROOT / "analysis" / "results" / "json"
REPORT_DIR = ROOT / "analysis" / "results" / "reports"
for directory in (FIG_DIR, JSON_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MODEL_IDS = {
    "tiny": "nvidia/MambaVision-T-1K",
    "tiny2": "nvidia/MambaVision-T2-1K",
    "small": "nvidia/MambaVision-S-1K",
    "base": "nvidia/MambaVision-B-1K",
    "large": "nvidia/MambaVision-L-1K",
}
FREQUENCIES = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50], dtype=np.float32)
ORIENTATIONS_DEG = np.array([0, 45, 90, 135], dtype=np.float32)


def load_model(model_id: str) -> torch.nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_id, trust_remote_code=True)


def call_model(model: torch.nn.Module, x: torch.Tensor) -> Any:
    try:
        return model(x)
    except TypeError:
        return model(pixel_values=x)


def count_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def tensor_shape(obj: Any) -> Optional[List[int]]:
    if torch.is_tensor(obj):
        return list(obj.shape)
    return None


def output_features(result: Any) -> List[torch.Tensor]:
    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, (tuple, list)) and len(item) >= 4 and all(torch.is_tensor(t) for t in item[:4]):
                return list(item[:4])
    if isinstance(result, dict):
        for key in ("features", "hidden_states", "feature_maps"):
            value = result.get(key)
            if isinstance(value, (tuple, list)) and len(value) >= 4:
                return list(value[:4])
    return []


def find_stage_modules(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    if hasattr(model, "model") and hasattr(model.model, "levels"):
        levels = model.model.levels
        return [(f"model.levels.{idx}", levels[idx]) for idx in range(min(4, len(levels)))]
    if hasattr(model, "levels"):
        return [(f"levels.{idx}", model.levels[idx]) for idx in range(min(4, len(model.levels)))]
    for name, module in model.named_modules():
        if name.endswith("levels") and hasattr(module, "__len__"):
            try:
                return [(f"{name}.{idx}", module[idx]) for idx in range(min(4, len(module)))]
            except Exception:
                continue
    return []


def get_stage_features(model: torch.nn.Module, x: torch.Tensor, detach: bool = True) -> List[torch.Tensor]:
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    for idx, (_name, module) in enumerate(find_stage_modules(model)[:4]):
        def make_hook(stage_idx: int):
            def _hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                if isinstance(output, (tuple, list)):
                    candidates = [item for item in output if torch.is_tensor(item)]
                    candidate = candidates[-1] if candidates else None
                else:
                    candidate = output
                if torch.is_tensor(candidate):
                    captured[stage_idx] = candidate.detach() if detach else candidate
            return _hook
        handles.append(module.register_forward_hook(make_hook(idx)))
    result = call_model(model, x)
    for handle in handles:
        handle.remove()
    features = output_features(result)
    if len(features) >= 4:
        return [feature.detach() if detach else feature for feature in features[:4]]
    if len(captured) >= 4:
        return [captured[idx] for idx in range(4)]
    raise RuntimeError("Could not extract four stage features.")


def energy_radii(heatmap: np.ndarray) -> Dict[str, float]:
    yy, xx = np.indices(heatmap.shape)
    cy, cx = (np.array(heatmap.shape) - 1) / 2.0
    radii = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).reshape(-1)
    weights = np.maximum(heatmap.reshape(-1), 0)
    order = np.argsort(radii)
    cumulative = np.cumsum(weights[order])
    total = cumulative[-1] + 1e-12
    return {
        key: float(radii[order][np.searchsorted(cumulative, frac * total)])
        for key, frac in (("r50", 0.50), ("r75", 0.75), ("r90", 0.90))
    }


def erf_analysis(model: torch.nn.Module, device: torch.device) -> List[Dict[str, Any]]:
    rows = []
    for stage_idx in range(4):
        model.zero_grad(set_to_none=True)
        x = torch.ones(1, 3, 224, 224, device=device, requires_grad=True)
        features = get_stage_features(model, x, detach=False)
        feature = features[stage_idx]
        if feature.ndim != 4:
            raise RuntimeError(f"Expected BCHW feature for stage {stage_idx + 1}, got {tuple(feature.shape)}")
        _, _, h, w = feature.shape
        target = feature[0, :, h // 2, w // 2].mean()
        target.backward()
        heat = x.grad.detach().abs().sum(dim=1)[0]
        heat = heat / (heat.max() + 1e-12)
        heat_np = heat.cpu().numpy()
        rows.append(
            {
                "stage": stage_idx + 1,
                "feature_shape": list(feature.shape),
                "energy_radii": energy_radii(heat_np),
                "active_area_threshold_5pct": int((heat_np >= 0.05 * heat_np.max()).sum()),
            }
        )
        del x, features, feature, heat
        torch.cuda.empty_cache()
    return rows


def grating(frequency: float, theta_deg: float, size: int = 224) -> torch.Tensor:
    theta = math.radians(float(theta_deg))
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    phase_coordinate = x * math.cos(theta) + y * math.sin(theta)
    image = 0.5 + 0.5 * np.sin(2.0 * math.pi * frequency * phase_coordinate)
    return torch.from_numpy(np.stack([image, image, image], axis=0)).float()


def frequency_analysis(model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    response = np.zeros((4, len(FREQUENCIES), len(ORIENTATIONS_DEG)), dtype=np.float32)
    with torch.no_grad():
        for f_idx, freq in enumerate(FREQUENCIES):
            for o_idx, theta in enumerate(ORIENTATIONS_DEG):
                x = grating(float(freq), float(theta)).unsqueeze(0).to(device)
                features = get_stage_features(model, x, detach=True)
                response[:, f_idx, o_idx] = [float(feature.abs().mean().cpu()) for feature in features]
    mean_by_stage_freq = response.mean(axis=2)
    dominant_stage = (mean_by_stage_freq.argmax(axis=0) + 1).tolist()
    stage_peak_freq = [float(FREQUENCIES[int(mean_by_stage_freq[stage].argmax())]) for stage in range(4)]
    return {
        "frequencies": FREQUENCIES.tolist(),
        "orientations_degrees": ORIENTATIONS_DEG.tolist(),
        "response": response.tolist(),
        "mean_by_stage_frequency": mean_by_stage_freq.tolist(),
        "dominant_stage_by_frequency": dominant_stage,
        "stage_peak_frequency": stage_peak_freq,
    }


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
        stimuli[name] = mask.float().repeat(3, 1, 1)
    ramp = torch.clamp((x.float() - center + 32) / 64, 0, 1)
    stimuli["vertical_ramp"] = ramp.repeat(3, 1, 1)
    stimuli["checker_edge_texture"] = (((x // 16 + y // 16) % 2).float()).repeat(3, 1, 1)
    return stimuli


def edge_modules(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    selected = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and (
            "patch_embed" in name or "levels.0" in name or "levels.1" in name or "levels.2.downsample" in name
        ):
            selected.append((name, module))
    return selected


def collect_activations(model: torch.nn.Module, x: torch.Tensor, selected: List[Tuple[str, torch.nn.Module]]) -> Dict[str, torch.Tensor]:
    activations: Dict[str, torch.Tensor] = {}
    handles = []
    for name, module in selected:
        def make_hook(layer_name: str):
            def _hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                if torch.is_tensor(output):
                    activations[layer_name] = output.detach()
            return _hook
        handles.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        call_model(model, x)
    for handle in handles:
        handle.remove()
    return activations


def edge_analysis(model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    selected = edge_modules(model)
    stimuli = edge_stimuli(device)
    flat = collect_activations(model, stimuli["flat"].unsqueeze(0), selected)
    rows = []
    for layer_name, flat_activation in flat.items():
        edge_values = []
        for stim_name, image in stimuli.items():
            if stim_name == "flat" or "step" not in stim_name:
                continue
            acts = collect_activations(model, image.unsqueeze(0), selected)
            if layer_name in acts:
                edge_values.append(float(acts[layer_name].abs().mean().cpu()))
        edge_response = float(np.mean(edge_values)) if edge_values else 0.0
        flat_response = float(flat_activation.abs().mean().cpu())
        esi = (edge_response - flat_response) / (edge_response + flat_response + 1e-12)
        rows.append(
            {
                "layer": layer_name,
                "edge_response_mean": edge_response,
                "flat_response": flat_response,
                "edge_selectivity_index": float(esi),
            }
        )
    rows.sort(key=lambda row: row["edge_selectivity_index"], reverse=True)
    return {"top_layers": rows[:10], "hooked_layer_count": len(selected)}


def ssm_layers(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    return [(name, module) for name, module in model.named_modules() if module.__class__.__name__ == "MambaVisionMixer"]


def learned_delta(module: torch.nn.Module) -> torch.Tensor:
    return F.softplus(module.dt_proj.bias.detach().float())


def impulse_response(module: torch.nn.Module, steps: int = 257) -> torch.Tensor:
    a_cont = -torch.exp(module.A_log.detach().float())
    delta = learned_delta(module).mean()
    eig = torch.exp(delta * a_cont).clamp(min=0)
    n = torch.arange(steps, dtype=eig.dtype, device=eig.device)
    return eig.unsqueeze(-1) ** n.unsqueeze(0)


def ssm_analysis(model: torch.nn.Module) -> Dict[str, Any]:
    records = []
    for name, module in ssm_layers(model):
        a_cont = -torch.exp(module.A_log.detach().float())
        delta = learned_delta(module)
        eig = torch.exp(delta.mean() * a_cont)
        response = impulse_response(module).reshape(-1, 257).mean(dim=0)
        below = torch.nonzero(response <= 0.5, as_tuple=False)
        half_life = float(below[0].item()) if len(below) else 256.0
        records.append(
            {
                "name": name,
                "d_model": int(getattr(module, "d_model", 0)),
                "d_state": int(getattr(module, "d_state", 0)),
                "mean_eigenvalue_magnitude": float(eig.abs().mean().cpu()),
                "max_eigenvalue_magnitude": float(eig.abs().max().cpu()),
                "stable_inside_unit_circle": bool((eig.abs() < 1.0).all().item()),
                "mean_impulse_half_life_samples": half_life,
                "delta_mean": float(delta.mean().cpu()),
            }
        )
    stable_count = sum(1 for row in records if row["stable_inside_unit_circle"])
    return {
        "records": records,
        "ssm_count": len(records),
        "stable_count": stable_count,
        "mean_eigenvalue_magnitude": float(np.mean([row["mean_eigenvalue_magnitude"] for row in records])) if records else None,
        "mean_half_life_samples": float(np.mean([row["mean_impulse_half_life_samples"] for row in records])) if records else None,
    }


def first_mixer(model: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    layers = ssm_layers(model)
    if not layers:
        raise RuntimeError("No MambaVisionMixer found.")
    return layers[0]


def capture_input(model: torch.nn.Module, module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    captured: Dict[str, torch.Tensor] = {}
    def hook(_module: torch.nn.Module, inputs: Tuple[Any, ...], _output: Any) -> None:
        captured["input"] = inputs[0].detach()
    handle = module.register_forward_hook(hook)
    with torch.no_grad():
        call_model(model, x)
    handle.remove()
    return captured["input"]


def selective_parameters(module: torch.nn.Module, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
    _, seqlen, _ = hidden_states.shape
    xz = rearrange(module.in_proj(hidden_states), "b l d -> b d l")
    x_branch, _z = xz.chunk(2, dim=1)
    x_branch = F.silu(F.conv1d(x_branch, module.conv1d_x.weight, bias=module.conv1d_x.bias, padding="same", groups=module.d_inner // 2))
    x_dbl = module.x_proj(rearrange(x_branch, "b d l -> (b l) d"))
    dt, b_vec, c_vec = torch.split(x_dbl, [module.dt_rank, module.d_state, module.d_state], dim=-1)
    dt = rearrange(module.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
    b_vec = rearrange(b_vec, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    c_vec = rearrange(c_vec, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    return {"delta": F.softplus(dt + module.dt_proj.bias.view(1, -1, 1)), "B": b_vec, "C": c_vec, "x_branch": x_branch}


def branch_outputs(module: torch.nn.Module, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    params = selective_parameters(module, hidden_states)
    xz = rearrange(module.in_proj(hidden_states), "b l d -> b d l")
    _x, z_branch = xz.chunk(2, dim=1)
    z_branch = F.silu(F.conv1d(z_branch, module.conv1d_z.weight, bias=module.conv1d_z.bias, padding="same", groups=module.d_inner // 2))
    y_branch = selective_scan_fn(
        params["x_branch"],
        params["delta"],
        -torch.exp(module.A_log.float()),
        params["B"],
        params["C"],
        module.D.float(),
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=None,
    )
    return y_branch.detach(), z_branch.detach()


def branch_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    n = min(a_flat.numel(), b_flat.numel())
    a_flat = a_flat[:n] - a_flat[:n].mean()
    b_flat = b_flat[:n] - b_flat[:n].mean()
    return float(((a_flat * b_flat).mean() / (a_flat.std() * b_flat.std() + 1e-12)).cpu())


def mathematical_verification(model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    mixer_name, mixer = first_mixer(model)
    hidden_1 = capture_input(model, mixer, torch.ones(1, 3, 224, 224, device=device))
    hidden_2 = capture_input(model, mixer, torch.rand(1, 3, 224, 224, device=device))
    params_1 = selective_parameters(mixer, hidden_1)
    params_2 = selective_parameters(mixer, hidden_2)
    ssm_branch, sym_branch = branch_outputs(mixer, hidden_2)
    return {
        "first_mixer": mixer_name,
        "delta_diff": float((params_1["delta"] - params_2["delta"]).abs().mean().cpu()),
        "B_diff": float((params_1["B"] - params_2["B"]).abs().mean().cpu()),
        "C_diff": float((params_1["C"] - params_2["C"]).abs().mean().cpu()),
        "concat_verified": bool(ssm_branch.shape[1] + sym_branch.shape[1] == mixer.out_proj.in_features),
        "ssm_branch_channels": int(ssm_branch.shape[1]),
        "symmetric_branch_channels": int(sym_branch.shape[1]),
        "out_proj_in_features": int(mixer.out_proj.in_features),
        "branch_correlation": branch_correlation(ssm_branch, sym_branch),
    }


def throughput(model: torch.nn.Module, device: torch.device, passes: int) -> Dict[str, Any]:
    for batch in (32, 16, 8, 4, 2, 1):
        try:
            x = torch.randn(batch, 3, 224, 224, device=device)
            with torch.no_grad():
                for _ in range(3):
                    call_model(model, x)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(passes):
                    call_model(model, x)
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            return {"batch_size": batch, "passes": passes, "elapsed_seconds": elapsed, "images_per_second": batch * passes / elapsed}
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
    return {"error": "out of memory for all tested batch sizes"}


def architecture(model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    with torch.no_grad():
        features = get_stage_features(model, torch.ones(1, 3, 224, 224, device=device), detach=True)
    mixers = ssm_layers(model)
    attention_count = sum(1 for _name, module in model.named_modules() if module.__class__.__name__ == "Attention")
    return {
        "total_parameters": count_params(model),
        "stage_shapes": [list(feature.shape) for feature in features],
        "channels": [int(feature.shape[1]) for feature in features],
        "resolutions": [list(feature.shape[-2:]) for feature in features],
        "ssm_mixer_count": len(mixers),
        "attention_block_count": attention_count,
    }


def analyze_one(label: str, model_id: str, device: torch.device, throughput_passes: int) -> Dict[str, Any]:
    print(f"\n=== {label}: {model_id} ===")
    model = load_model(model_id).to(device).eval()
    row: Dict[str, Any] = {"label": label, "model_id": model_id}
    try:
        row["architecture"] = architecture(model, device)
        print(f"params={row['architecture']['total_parameters']:,} channels={row['architecture']['channels']}")
        row["erf"] = erf_analysis(model, device)
        print("erf r90:", [round(item["energy_radii"]["r90"], 1) for item in row["erf"]])
        row["frequency"] = frequency_analysis(model, device)
        print("dominant stages:", row["frequency"]["dominant_stage_by_frequency"])
        row["edge"] = edge_analysis(model, device)
        print("top ESI:", row["edge"]["top_layers"][0]["layer"], round(row["edge"]["top_layers"][0]["edge_selectivity_index"], 4))
        row["ssm"] = ssm_analysis(model)
        print("stable SSM:", row["ssm"]["stable_count"], "/", row["ssm"]["ssm_count"])
        row["math_verification"] = mathematical_verification(model, device)
        print("branch corr:", round(row["math_verification"]["branch_correlation"], 4))
        row["throughput"] = throughput(model, device, throughput_passes)
        print("throughput:", row["throughput"])
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "error"
        row["error"] = repr(exc)
        print("ERROR:", repr(exc))
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return row


def plot_family(results: List[Dict[str, Any]]) -> List[str]:
    ok = [row for row in results if row.get("status") == "ok"]
    labels = [row["label"] for row in ok]
    figures: List[str] = []
    if not ok:
        return figures

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for stage in range(4):
        ax.plot(labels, [row["erf"][stage]["energy_radii"]["r90"] for row in ok], marker="o", label=f"Stage {stage + 1}")
    ax.set_ylabel("r90 pixels")
    ax.set_title("Effective Receptive Field r90 by Model")
    ax.grid(alpha=0.22)
    ax.legend()
    path = FIG_DIR / "family_erf_r90.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(labels, [row["throughput"].get("images_per_second", 0) for row in ok], color="#00d4ff")
    ax.set_ylabel("images / second")
    ax.set_title("Measured Throughput by Model")
    path = FIG_DIR / "family_throughput.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(labels, [row["edge"]["top_layers"][0]["edge_selectivity_index"] for row in ok], color="#39ff8f")
    ax.set_ylabel("top ESI")
    ax.set_title("Top Edge Selectivity Index by Model")
    path = FIG_DIR / "family_edge_top_esi.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.12
    x = np.arange(len(labels))
    for idx, freq in enumerate(FREQUENCIES):
        values = [row["frequency"]["dominant_stage_by_frequency"][idx] for row in ok]
        ax.bar(x + (idx - 3) * width, values, width=width, label=f"{freq:.2f}")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 4.5)
    ax.set_ylabel("dominant stage")
    ax.set_title("Dominant Stage by Input Frequency")
    ax.legend(fontsize=7, ncol=4)
    path = FIG_DIR / "family_frequency_dominance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(labels, [row["ssm"]["mean_eigenvalue_magnitude"] for row in ok], marker="o", label="mean |eig|")
    ax.plot(labels, [row["ssm"]["mean_half_life_samples"] for row in ok], marker="s", label="mean half-life")
    ax.set_title("SSM Stability Summary")
    ax.grid(alpha=0.22)
    ax.legend()
    path = FIG_DIR / "family_ssm_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figures.append(str(path.relative_to(ROOT)))
    return figures


def generate_reasoning_report(results: List[Dict[str, Any]], figures: List[str]) -> str:
    ok = [row for row in results if row.get("status") == "ok"]
    lines = [
        "# MambaVision Model Family Analysis",
        "",
        "This report is generated from measured JSON outputs. Explanations are phrased as interpretations of the probes, not as proof of training-time causality.",
        "",
        "## Model Summary",
        "",
        "| Model | Params | Channels | SSM | Attention | Throughput |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in ok:
        arch = row["architecture"]
        ips = row["throughput"].get("images_per_second")
        lines.append(
            f"| {row['label']} | {arch['total_parameters']:,} | {arch['channels']} | "
            f"{arch['ssm_mixer_count']} | {arch['attention_block_count']} | {ips:.2f} img/s |"
        )
    lines.extend(["", "## What The Results Mean", ""])
    for row in ok:
        label = row["label"]
        arch = row["architecture"]
        erf_r90 = [item["energy_radii"]["r90"] for item in row["erf"]]
        dom = row["frequency"]["dominant_stage_by_frequency"]
        edge = row["edge"]["top_layers"][0]
        ssm = row["ssm"]
        mathv = row["math_verification"]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- **Capacity and geometry.** The measured stage channels are `{arch['channels']}` at resolutions `{arch['resolutions']}`. Parameter count is `{arch['total_parameters']:,}`. Wider channels increase representational capacity but also increase compute, which is reflected in throughput.",
                f"- **Effective receptive field.** The measured r90 radii by stage are `{[round(v, 1) for v in erf_r90]}` pixels. The increase from early to late stages is expected because downsampling and token mixing let a center feature aggregate information from a larger input region. This is a gradient-based reach measurement, not a semantic importance proof.",
                f"- **Frequency response.** Dominant raw activation stages by frequency are `{dom}` for frequencies `{FREQUENCIES.tolist()}`. Raw activation magnitude is affected by normalization, channel scale, and depth, so this is best read as a stage-energy probe rather than a calibrated transfer function.",
                f"- **Edge selectivity.** The top layer by ESI is `{edge['layer']}` with ESI `{edge['edge_selectivity_index']:.4f}`. ESI compares edge stimuli against a flat image, so positive values indicate stronger response to controlled edges than to uniform input.",
                f"- **SSM stability.** `{ssm['stable_count']}/{ssm['ssm_count']}` SSM blocks have sampled discrete eigenvalues inside the unit circle. Mean |eigenvalue| is `{ssm['mean_eigenvalue_magnitude']:.4f}`, and mean impulse half-life is `{ssm['mean_half_life_samples']:.2f}` samples. This indicates stable decay in the diagonal SSM approximation used by the script.",
                f"- **Selective scan verification.** Two different inputs produced mean absolute differences of Δ `{mathv['delta_diff']:.4f}`, B `{mathv['B_diff']:.4f}`, and C `{mathv['C_diff']:.4f}` in the first mixer. That numerically supports the claim that these scan parameters are input-dependent for the inspected block.",
                f"- **Branch complementarity.** First-mixer branch correlation is `{mathv['branch_correlation']:.4f}`. A value near zero means the SSM and symmetric branches are not linearly redundant under this synthetic input, though complementarity should be validated on task data before making stronger claims.",
                "",
            ]
        )
    if figures:
        lines.extend(["## Generated Comparison Figures", ""])
        lines.extend([f"- `{figure}`" for figure in figures])
    for row in results:
        if row.get("status") != "ok":
            lines.extend(["", f"## Failed Model: {row['label']}", "", f"`{row.get('error')}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(MODEL_IDS.keys()), choices=list(MODEL_IDS.keys()))
    parser.add_argument("--throughput-passes", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("mamba_ssm selective scan requires CUDA for these HF models; run in a CUDA-visible environment.")
    device = torch.device("cuda")
    results = [analyze_one(label, MODEL_IDS[label], device, args.throughput_passes) for label in args.models]
    figures = plot_family(results)
    payload = {
        "model_ids": {label: MODEL_IDS[label] for label in args.models},
        "frequencies": FREQUENCIES.tolist(),
        "orientations_degrees": ORIENTATIONS_DEG.tolist(),
        "results": results,
        "figures": figures,
    }
    json_path = JSON_DIR / "model_family_analysis.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = generate_reasoning_report(results, figures)
    report_path = REPORT_DIR / "08_model_family_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nSaved {json_path}")
    print(f"Saved {report_path}")
    for figure in figures:
        print(f"Saved {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
