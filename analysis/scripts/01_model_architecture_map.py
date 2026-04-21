#!/usr/bin/env python3
"""
Build a recursive architecture map for the HuggingFace MambaVision-T-1K model.

The script uses a dummy 224x224 input only to discover tensor shapes. Parameter
counts and block labels are computed directly from the module tree.
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
import torch


plt.style.use("dark_background")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "analysis" / "results" / "figures"
JSON_DIR = ROOT / "analysis" / "results" / "json"
REPORT_DIR = ROOT / "analysis" / "results" / "reports"
for directory in (FIG_DIR, JSON_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_model() -> torch.nn.Module:
    """Load the exact HF checkpoint requested by the analysis spec."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "nvidia/MambaVision-T-1K",
        trust_remote_code=True,
    )
    return model


def tensor_shape(obj: Any) -> Optional[Any]:
    """Return a JSON-friendly shape description for tensors/nested outputs."""
    if torch.is_tensor(obj):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [tensor_shape(item) for item in obj if tensor_shape(item) is not None]
    if isinstance(obj, dict):
        return {key: tensor_shape(value) for key, value in obj.items() if tensor_shape(value) is not None}
    return None


def call_model(model: torch.nn.Module, x: torch.Tensor) -> Any:
    """Handle both plain PyTorch and HF-style forward signatures."""
    try:
        return model(x)
    except TypeError:
        return model(pixel_values=x)


def classify_module(name: str, module: torch.nn.Module) -> str:
    """Classify the architectural role from stable class/name signals."""
    cls = module.__class__.__name__.lower()
    lname = name.lower()
    if "mambavisionmixer" in cls or "selective" in cls or "ssm" in lname or "a_log" in lname:
        return "SSM block"
    if "attention" in cls or ".attn" in lname:
        return "Attention block"
    if "conv" in cls or "conv" in lname or "patch" in cls or "downsample" in cls:
        return "CNN block"
    if "mlp" in cls or "ffn" in lname:
        return "MLP/FFN"
    if "norm" in cls or "batchnorm" in cls or "layernorm" in cls:
        return "Normalization"
    return "Other"


def count_params(module: torch.nn.Module, recurse: bool = True) -> int:
    return sum(param.numel() for param in module.parameters(recurse=recurse))


def find_stage_modules(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    """Return the four hierarchical stage modules when the HF wrapper exposes them."""
    candidates: List[Tuple[str, torch.nn.Module]] = []
    for prefix in ("levels", "model.levels", "backbone.levels", "mambavision.levels"):
        obj: Any = model
        ok = True
        for part in prefix.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and hasattr(obj, "__len__"):
            try:
                return [(f"{prefix}.{idx}", obj[idx]) for idx in range(len(obj))]
            except Exception:
                pass
    for name, module in model.named_modules():
        if name.endswith("levels") and hasattr(module, "__len__"):
            try:
                candidates = [(f"{name}.{idx}", module[idx]) for idx in range(len(module))]
                if len(candidates) >= 4:
                    return candidates[:4]
            except Exception:
                continue
    return candidates[:4]


def capture_shapes(model: torch.nn.Module, device: torch.device) -> Dict[str, Dict[str, Any]]:
    """Register lightweight hooks and run one synthetic input to record shapes."""
    shapes: Dict[str, Dict[str, Any]] = {}
    handles = []

    def hook(name: str):
        def _hook(_module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            shapes[name] = {
                "input_shape": tensor_shape(inputs[0]) if inputs else None,
                "output_shape": tensor_shape(output),
            }

        return _hook

    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(hook(name)))
    for name, module in find_stage_modules(model):
        handles.append(module.register_forward_hook(hook(name)))

    model.eval()
    with torch.no_grad():
        dummy = torch.ones(1, 3, 224, 224, device=device)
        call_model(model, dummy)

    for handle in handles:
        handle.remove()
    return shapes


def build_graph(model: torch.nn.Module, shapes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    total = count_params(model)
    graph = []
    for name, module in model.named_modules():
        display_name = name or "<root>"
        local_params = count_params(module, recurse=False)
        recursive_params = count_params(module, recurse=True)
        graph.append(
            {
                "name": display_name,
                "class": module.__class__.__name__,
                "local_parameter_count": local_params,
                "recursive_parameter_count": recursive_params,
                "percent_of_total_parameters": round(100.0 * recursive_params / total, 6) if total else 0.0,
                "block_type": classify_module(name, module),
                "input_shape": shapes.get(name, {}).get("input_shape"),
                "output_shape": shapes.get(name, {}).get("output_shape"),
                "children": list(module._modules.keys()),
            }
        )
    return graph


def print_tree(graph: List[Dict[str, Any]], total_params: int) -> None:
    print("\nRecursive module tree with tensor shapes")
    print("=" * 110)
    print(f"Total parameters: {total_params:,}")
    print(f"{'module':58s} {'class':28s} {'in -> out':36s} {'params':>14s} {'%':>8s}")
    print("-" * 150)
    for item in graph:
        name = item["name"]
        indent = "  " * (0 if name == "<root>" else name.count(".") + 1)
        label = f"{indent}{name.split('.')[-1] if name != '<root>' else '<root>'}"
        shape_text = f"{item['input_shape']} -> {item['output_shape']}"
        print(
            f"{label[:58]:58s} {item['class'][:28]:28s} "
            f"{shape_text[:36]:36s} {item['recursive_parameter_count']:14,d} "
            f"{item['percent_of_total_parameters']:7.3f}%"
        )


def stage_specs(model: torch.nn.Module, shapes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    specs = []
    stages = find_stage_modules(model)
    for idx, (name, module) in enumerate(stages[:4], start=1):
        out_shape = shapes.get(name, {}).get("output_shape")
        channels = None
        resolution = None
        if isinstance(out_shape, list) and len(out_shape) == 4:
            channels = out_shape[1]
            resolution = f"{out_shape[2]}x{out_shape[3]}"
        if channels is None and hasattr(module, "blocks"):
            # Infer from the first parameter-bearing child when no hook shape was captured.
            for param in module.parameters():
                if param.ndim > 0:
                    channels = int(param.shape[0])
                    break
        role = "Conv" if getattr(module, "conv", idx < 3) else "Mamba+Attn"
        specs.append(
            {
                "stage": idx,
                "name": name,
                "role": role,
                "channels": channels,
                "resolution": resolution or "unknown",
                "depth": len(getattr(module, "blocks", [])) if hasattr(module, "blocks") else None,
                "window_size": getattr(module, "window_size", None),
            }
        )
    return specs


def draw_stage_diagram(specs: List[Dict[str, Any]]) -> Path:
    if not specs:
        specs = [
            {"stage": 1, "role": "Conv", "channels": 80, "resolution": "56x56", "depth": 1},
            {"stage": 2, "role": "Conv", "channels": 160, "resolution": "28x28", "depth": 3},
            {"stage": 3, "role": "Mamba+Attn", "channels": 320, "resolution": "14x14", "depth": 8},
            {"stage": 4, "role": "Mamba+Attn", "channels": 640, "resolution": "7x7", "depth": 4},
        ]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_axis_off()
    colors = ["#00d4ff", "#39ff8f", "#7b61ff", "#ffcc66"]
    x_positions = [0.08, 0.32, 0.56, 0.80]
    for idx, spec in enumerate(specs[:4]):
        x0 = x_positions[idx]
        rect = plt.Rectangle((x0, 0.32), 0.16, 0.34, facecolor="#0d1620", edgecolor=colors[idx], linewidth=2.5)
        ax.add_patch(rect)
        title = f"Stage {spec['stage']}: {spec['role']}"
        depth = f"depth x{spec.get('depth')}" if spec.get("depth") is not None else "depth ?"
        channels = f"C={spec.get('channels') or '?'}"
        res = f"R={spec.get('resolution')}"
        ax.text(x0 + 0.08, 0.56, title, ha="center", va="center", fontsize=13, color="#e8edf3", weight="bold")
        ax.text(x0 + 0.08, 0.45, f"{channels} | {res}", ha="center", va="center", fontsize=11, color="#00d4ff")
        ax.text(x0 + 0.08, 0.36, depth, ha="center", va="center", fontsize=10, color="#6b7f94")
        if idx < 3:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.02, 0.49),
                xytext=(x0 + 0.18, 0.49),
                arrowprops=dict(arrowstyle="->", color="#e8edf3", lw=2),
            )
    ax.text(0.5, 0.82, "MambaVision-T-1K Hierarchical Stage Diagram", ha="center", fontsize=16, color="#e8edf3")
    ax.text(0.5, 0.18, "Spatial resolution is measured from a 224x224 synthetic input.", ha="center", fontsize=10, color="#6b7f94")
    out_path = FIG_DIR / "mambavision_stage_diagram.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def nearest_parent(model: torch.nn.Module, child_name: str) -> Optional[torch.nn.Module]:
    parts = child_name.split(".")[:-1]
    while parts:
        parent_name = ".".join(parts)
        parent = dict(model.named_modules()).get(parent_name)
        if parent is not None and hasattr(parent, "mlp"):
            return parent
        parts.pop()
    return None


def print_mixer_and_attention_details(model: torch.nn.Module) -> Dict[str, Any]:
    details: Dict[str, Any] = {"mixers": [], "attention_blocks": []}
    modules = dict(model.named_modules())
    print("\nMambaVision Mixer blocks")
    print("=" * 110)
    for name, module in modules.items():
        if module.__class__.__name__ == "MambaVisionMixer":
            parent = nearest_parent(model, name)
            entry = {
                "name": name,
                "d_model": getattr(module, "d_model", None),
                "d_state": getattr(module, "d_state", None),
                "d_conv": getattr(module, "d_conv", None),
                "ssm_branch_components": ["in_proj half x", "conv1d_x", "x_proj", "dt_proj", "A_log", "D", "selective_scan_fn"],
                "non_ssm_symmetric_branch_components": ["in_proj half z", "conv1d_z", "SiLU"],
                "mlp_ffn_components": list(getattr(parent, "mlp", torch.nn.Identity())._modules.keys()) if parent is not None else [],
                "fusion": "torch.cat([selective_scan_output, symmetric_branch_output], dim=1) followed by out_proj",
            }
            details["mixers"].append(entry)
            print(f"- {name}: d_model={entry['d_model']}, d_state={entry['d_state']}, d_conv={entry['d_conv']}")
            print(f"  SSM branch: {', '.join(entry['ssm_branch_components'])}")
            print(f"  non-SSM branch: {', '.join(entry['non_ssm_symmetric_branch_components'])}")
            print(f"  MLP/FFN: {entry['mlp_ffn_components'] or 'not found'}")
            print(f"  fusion: {entry['fusion']}")

    print("\nTransformer self-attention blocks")
    print("=" * 110)
    for name, module in modules.items():
        if module.__class__.__name__ == "Attention":
            window_size = None
            prefix = name.split(".blocks.")[0] if ".blocks." in name else ""
            stage_module = modules.get(prefix)
            if stage_module is not None:
                window_size = getattr(stage_module, "window_size", None)
            embed_dim = None
            qkv = getattr(module, "qkv", None)
            if qkv is not None and hasattr(qkv, "in_features"):
                embed_dim = qkv.in_features
            entry = {
                "name": name,
                "window_size": window_size,
                "num_heads": getattr(module, "num_heads", None),
                "embedding_dimension": embed_dim,
            }
            details["attention_blocks"].append(entry)
            print(
                f"- {name}: window_size={entry['window_size']}, "
                f"heads={entry['num_heads']}, embed_dim={entry['embedding_dimension']}"
            )
    return details


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading nvidia/MambaVision-T-1K on {device}...")
    model = load_model().to(device)

    print("Running one 224x224 forward pass for shape discovery...")
    shapes = capture_shapes(model, device)
    total_params = count_params(model)
    graph = build_graph(model, shapes)
    specs = stage_specs(model, shapes)
    block_details = print_mixer_and_attention_details(model)

    print_tree(graph, total_params)

    json_path = JSON_DIR / "architecture_graph.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": "nvidia/MambaVision-T-1K",
                "device": str(device),
                "total_parameters": total_params,
                "stages": specs,
                "blocks": block_details,
                "modules": graph,
            },
            f,
            indent=2,
        )

    diagram_path = draw_stage_diagram(specs)
    report_path = REPORT_DIR / "01_model_architecture_map.md"
    report_path.write_text(
        "\n".join(
            [
                "# MambaVision Architecture Map",
                "",
                f"- Model: `nvidia/MambaVision-T-1K`",
                f"- Total parameters: `{total_params:,}`",
                f"- JSON: `{json_path.relative_to(ROOT)}`",
                f"- Stage diagram: `{diagram_path.relative_to(ROOT)}`",
                "",
                "## Stages",
                "",
                *[
                    f"- Stage {s['stage']}: {s['role']}, C={s.get('channels')}, R={s.get('resolution')}, depth={s.get('depth')}"
                    for s in specs
                ],
            ]
        ),
        encoding="utf-8",
    )

    print("\nSaved outputs")
    print(f"- JSON architecture graph: {json_path}")
    print(f"- Stage diagram: {diagram_path}")
    print(f"- Markdown report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
