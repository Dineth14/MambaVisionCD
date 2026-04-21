#!/usr/bin/env python3
"""
Layer activation visualizer for MambaVision-T-1K.

This script uses five deterministic inputs and records representative behavior
from stem convolutions, downsampling layers, MambaVision mixer branches,
self-attention blocks, MLPs, and final average-pooled features.
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
from einops import rearrange


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


def make_test_images(device: torch.device, size: int = 224) -> Dict[str, torch.Tensor]:
    y, x = torch.meshgrid(torch.linspace(0, 1, size, device=device), torch.linspace(0, 1, size, device=device), indexing="ij")
    natural = torch.stack([0.35 + 0.4 * x, 0.25 + 0.5 * y, 0.55 - 0.25 * x + 0.1 * torch.sin(20 * y)], dim=0).clamp(0, 1)
    aerial = torch.stack(
        [
            0.34 + 0.18 * (((x * 8).floor() + (y * 6).floor()) % 2),
            0.38 + 0.25 * (torch.abs(x - 0.62) < 0.015).float(),
            0.32 + 0.35 * (torch.abs(y - 0.42 - 0.05 * torch.sin(9 * x)) < 0.015).float(),
        ],
        dim=0,
    ).clamp(0, 1)
    texture = (0.5 + 0.25 * torch.sin(45 * x) + 0.25 * torch.cos(37 * y)).repeat(3, 1, 1).clamp(0, 1)
    noise = torch.rand(3, size, size, device=device)
    blank = torch.full((3, size, size), 0.5, device=device)
    return {"natural": natural, "aerial": aerial, "texture": texture, "noise": noise, "blank": blank}


def first_module(model: torch.nn.Module, predicate) -> Tuple[str, torch.nn.Module]:
    for name, module in model.named_modules():
        if predicate(name, module):
            return name, module
    raise RuntimeError("Requested module not found.")


def capture_module_io(model: torch.nn.Module, module: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if inputs and torch.is_tensor(inputs[0]):
            captured["input"] = inputs[0].detach()
        candidate = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(candidate):
            captured["output"] = candidate.detach()

    handle = module.register_forward_hook(hook)
    with torch.no_grad():
        call_model(model, x)
    handle.remove()
    return captured["input"], captured["output"]


def plot_channel_grid(feature: torch.Tensor, title: str, out_name: str, channels: int = 64) -> Path:
    fmap = feature.detach().float()
    if fmap.ndim == 3:
        fmap = fmap.unsqueeze(0)
    if fmap.ndim == 4:
        fmap = fmap[0]
    elif fmap.ndim == 3:
        pass
    n = min(channels, fmap.shape[0])
    cols = 8
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.25))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx < n:
            ax.imshow(fmap[idx].cpu().numpy(), cmap="magma")
    fig.suptitle(title)
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def mixer_branch_outputs(module: torch.nn.Module, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute SSM and symmetric branches from a MambaVisionMixer input."""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    _, seqlen, _ = hidden_states.shape
    xz = module.in_proj(hidden_states)
    xz = rearrange(xz, "b l d -> b d l")
    x_branch, z_branch = xz.chunk(2, dim=1)
    a_matrix = -torch.exp(module.A_log.float())

    # The x path is the selective SSM branch. Its delta/B/C are input-dependent
    # because x_proj is applied to the current activation sequence.
    x_branch = F.silu(
        F.conv1d(
            input=x_branch,
            weight=module.conv1d_x.weight,
            bias=module.conv1d_x.bias,
            padding="same",
            groups=module.d_inner // 2,
        )
    )
    z_branch = F.silu(
        F.conv1d(
            input=z_branch,
            weight=module.conv1d_z.weight,
            bias=module.conv1d_z.bias,
            padding="same",
            groups=module.d_inner // 2,
        )
    )
    x_dbl = module.x_proj(rearrange(x_branch, "b d l -> (b l) d"))
    dt, b_vec, c_vec = torch.split(x_dbl, [module.dt_rank, module.d_state, module.d_state], dim=-1)
    dt = rearrange(module.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
    b_vec = rearrange(b_vec, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    c_vec = rearrange(c_vec, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    y_branch = selective_scan_fn(
        x_branch,
        dt,
        a_matrix,
        b_vec,
        c_vec,
        module.D.float(),
        z=None,
        delta_bias=module.dt_proj.bias.float(),
        delta_softplus=True,
        return_last_state=None,
    )
    return y_branch.detach(), z_branch.detach()


def plot_violin(before: torch.Tensor, after: torch.Tensor, title: str, out_name: str) -> Path:
    before_np = before.detach().float().flatten().cpu().numpy()
    after_np = after.detach().float().flatten().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.violinplot([before_np, after_np], showmeans=True, showextrema=True)
    ax.set_xticks([1, 2], ["before/selective input", "after/branch output"])
    ax.set_title(title)
    ax.grid(alpha=0.18)
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def correlation_and_variance(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a_flat = a.float().flatten(1)
    b_flat = b.float().flatten(1)
    min_dim = min(a_flat.shape[1], b_flat.shape[1])
    a_flat, b_flat = a_flat[:, :min_dim], b_flat[:, :min_dim]
    a_centered = a_flat - a_flat.mean(dim=1, keepdim=True)
    b_centered = b_flat - b_flat.mean(dim=1, keepdim=True)
    corr = (a_centered * b_centered).mean() / (a_centered.std() * b_centered.std() + 1e-12)
    return {
        "mean_feature_correlation": float(corr.cpu()),
        "ssm_variance": float(a_flat.var().cpu()),
        "symmetric_variance": float(b_flat.var().cpu()),
    }


def attention_maps(model: torch.nn.Module, x: torch.Tensor) -> List[Path]:
    paths = []
    seen_stages = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Attention":
            continue
        stage = 3 if "levels.2" in name else 4 if "levels.3" in name else len(seen_stages) + 3
        if stage in seen_stages:
            continue
        seen_stages.add(stage)
        captured_input, _ = capture_module_io(model, module, x)
        tokens = captured_input.detach()
        with torch.no_grad():
            qkv = module.qkv(tokens).reshape(tokens.shape[0], tokens.shape[1], 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _v = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            attn = (q * module.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)[0]
        side = int(math.sqrt(attn.shape[-1]))
        if side * side != attn.shape[-1]:
            continue
        center = attn.shape[-2] // 2
        for head in range(min(module.num_heads, 8)):
            fig, ax = plt.subplots(figsize=(4.4, 4.2))
            ax.imshow(attn[head, center].reshape(side, side).cpu().numpy(), cmap="viridis")
            ax.set_title(f"Stage {stage} Attention Head {head}")
            ax.set_xticks([])
            ax.set_yticks([])
            out = FIG_DIR / f"attention_maps_stage{stage}_head{head}.png"
            fig.savefig(out, dpi=220, bbox_inches="tight")
            plt.close(fig)
            paths.append(out)
    return paths


def avgpool_features(model: torch.nn.Module, images: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, List[str]]:
    vectors = []
    labels = []
    with torch.no_grad():
        for label, image in images.items():
            result = call_model(model, image.unsqueeze(0))
            if isinstance(result, (tuple, list)) and torch.is_tensor(result[0]):
                vec = result[0].detach().float().flatten().cpu().numpy()
            elif isinstance(result, dict) and "pooler_output" in result:
                vec = result["pooler_output"].detach().float().flatten().cpu().numpy()
            else:
                vec = torch.as_tensor(result).detach().float().flatten().cpu().numpy()
            vectors.append(vec)
            labels.append(label)
    matrix = np.stack(vectors)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, _vh = np.linalg.svd(matrix, full_matrices=False)
    coords = u[:, :2] * s[:2]
    return coords, labels


def plot_pca(coords: np.ndarray, labels: List[str]) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(coords[:, 0], coords[:, 1], s=80, color="#00d4ff")
    for (x_coord, y_coord), label in zip(coords, labels):
        ax.text(x_coord, y_coord, f" {label}", color="#e8edf3", va="center")
    ax.axhline(0, color="#6b7f94", lw=0.8, alpha=0.5)
    ax.axvline(0, color="#6b7f94", lw=0.8, alpha=0.5)
    ax.set_title("Final AvgPool Feature PCA")
    out = FIG_DIR / "avgpool_feature_pca.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def master_summary(paths: List[Path]) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_axis_off()
    labels = [
        "Stem Conv\nedges/textures",
        "Downsample\nresolution collapse",
        "SSM Branch\nselective sequence filter",
        "Symmetric Branch\nnon-causal local path",
        "Attention\ncenter-query maps",
        "MLP/FFN\nnonlinear channel mixing",
        "AvgPool\nsemantic aggregate",
    ]
    for idx, label in enumerate(labels):
        x = 0.04 + (idx % 4) * 0.24
        y = 0.58 if idx < 4 else 0.18
        rect = plt.Rectangle((x, y), 0.20, 0.24, facecolor="#0d1620", edgecolor="#00d4ff", linewidth=1.4)
        ax.add_patch(rect)
        ax.text(x + 0.10, y + 0.12, label, ha="center", va="center", color="#e8edf3", fontsize=11)
    ax.text(0.5, 0.93, "MambaVision Layer Function Summary", ha="center", color="#e8edf3", fontsize=16, weight="bold")
    out = FIG_DIR / "layer_function_summary.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading nvidia/MambaVision-T-1K on {device}...")
    model = load_model().to(device).eval()
    images = make_test_images(device)
    x = images["aerial"].unsqueeze(0)
    generated_paths: List[Path] = []

    stem_name, stem = first_module(model, lambda _n, m: isinstance(m, torch.nn.Conv2d))
    _, stem_output = capture_module_io(model, stem, x)
    generated_paths.append(plot_channel_grid(stem_output, f"Stem convolution activations: {stem_name}", "stem_activation_grid.png"))

    down_name, down = first_module(model, lambda n, _m: "downsample" in n.lower())
    down_in, down_out = capture_module_io(model, down, x)
    generated_paths.append(plot_channel_grid(down_in, f"Downsample input: {down_name}", "downsample_input_channels.png", 32))
    generated_paths.append(plot_channel_grid(down_out, f"Downsample output: {down_name}", "downsample_output_channels.png", 32))

    mixer_summaries = []
    for mixer_name, mixer in model.named_modules():
        if mixer.__class__.__name__ != "MambaVisionMixer":
            continue
        mixer_input, mixer_output = capture_module_io(model, mixer, x)
        ssm, sym = mixer_branch_outputs(mixer, mixer_input)
        generated_paths.append(plot_violin(mixer_input, ssm, f"SSM branch distribution: {mixer_name}", "ssm_branch_violin.png"))
        generated_paths.append(plot_violin(sym, torch.cat([ssm, sym], dim=1), f"Symmetric + concat: {mixer_name}", "symmetric_branch_violin.png"))
        generated_paths.append(plot_channel_grid(torch.cat([ssm, sym], dim=1), f"Branch concatenation: {mixer_name}", "branch_concatenation_map.png", 64))
        mixer_summaries.append({"name": mixer_name, **correlation_and_variance(ssm, sym)})
        break

    attention_paths = attention_maps(model, x)
    generated_paths.extend(attention_paths)

    mlp_name, mlp = first_module(model, lambda _n, m: m.__class__.__name__.lower() == "mlp")
    mlp_in, mlp_out = capture_module_io(model, mlp, x)
    generated_paths.append(plot_violin(mlp_in, mlp_out, f"MLP pre/post distribution: {mlp_name}", "mlp_pre_post_distribution.png"))

    coords, labels = avgpool_features(model, images)
    generated_paths.append(plot_pca(coords, labels))
    summary_path = master_summary(generated_paths)
    generated_paths.append(summary_path)

    json_path = JSON_DIR / "layer_activation_visualizer.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "test_images": list(images.keys()),
                "mixer_branch_summaries": mixer_summaries,
                "figures": [str(path.relative_to(ROOT)) for path in generated_paths],
            },
            f,
            indent=2,
        )

    (REPORT_DIR / "05_layer_activation_visualizer.md").write_text(
        "# Layer Activation Visualizer\n\n"
        f"- Master summary: `{summary_path.relative_to(ROOT)}`\n"
        f"- JSON: `{json_path.relative_to(ROOT)}`\n",
        encoding="utf-8",
    )

    print("\nLayer visualization summary")
    print("=" * 80)
    for summary in mixer_summaries:
        print(
            f"{summary['name']}: corr={summary['mean_feature_correlation']:.4f}, "
            f"var_ssm={summary['ssm_variance']:.4f}, var_sym={summary['symmetric_variance']:.4f}"
        )
    print(f"Saved {len(generated_paths)} figures and {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
