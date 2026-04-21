#!/usr/bin/env python3
"""
Numerically verify architectural claims for MambaVision-T-1K.

The checks are intentionally direct:
- Selective scan parameters delta/B/C are recomputed from two inputs.
- Regular convolution access is tested by future-token impulse influence.
- Concatenation fusion is checked from mixer dimensions and branch entropy.
- Stage resolutions are measured from a 224x224 input.
- Throughput is timed for 100 forward passes with batch-size fallback.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("HF_HOME", "/tmp/mambavision_hf_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mambavision_mpl_cache")

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "analysis" / "results" / "json"
REPORT_DIR = ROOT / "analysis" / "results" / "reports"
for directory in (JSON_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_model() -> torch.nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)


def call_model(model: torch.nn.Module, x: torch.Tensor) -> Any:
    try:
        return model(x)
    except TypeError:
        return model(pixel_values=x)


def first_mixer(model: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    for name, module in model.named_modules():
        if module.__class__.__name__ == "MambaVisionMixer":
            return name, module
    raise RuntimeError("No MambaVisionMixer found.")


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
    """Recompute input-dependent delta, B, and C from the SSM branch input."""
    _, seqlen, _ = hidden_states.shape
    xz = rearrange(module.in_proj(hidden_states), "b l d -> b d l")
    x_branch, _z = xz.chunk(2, dim=1)
    x_branch = F.silu(
        F.conv1d(
            x_branch,
            module.conv1d_x.weight,
            bias=module.conv1d_x.bias,
            padding="same",
            groups=module.d_inner // 2,
        )
    )
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
    z_branch = F.silu(
        F.conv1d(
            z_branch,
            module.conv1d_z.weight,
            bias=module.conv1d_z.bias,
            padding="same",
            groups=module.d_inner // 2,
        )
    )
    a_matrix = -torch.exp(module.A_log.float())
    y_branch = selective_scan_fn(
        params["x_branch"],
        params["delta"],
        a_matrix,
        params["B"],
        params["C"],
        module.D.float(),
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=None,
    )
    return y_branch.detach(), z_branch.detach()


def entropy(tensor: torch.Tensor, bins: int = 128) -> float:
    values = tensor.detach().float().flatten().cpu().numpy()
    hist, _ = np.histogram(values, bins=bins, density=True)
    probs = hist / (hist.sum() + 1e-12)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def future_token_access(conv: torch.nn.Conv1d, channels: int) -> Dict[str, float]:
    """If an impulse at token t+1 changes output at token t, the conv is non-causal."""
    length = 17
    center = length // 2
    x = torch.zeros(1, channels, length, device=conv.weight.device)
    x[:, :, center + 1] = 1.0
    with torch.no_grad():
        y = F.conv1d(x, conv.weight, bias=conv.bias, padding="same", groups=channels)
    return {
        "center_output_from_future_impulse_abs_mean": float(y[:, :, center].abs().mean().cpu()),
        "future_access_detected": bool(y[:, :, center].abs().mean().item() > 1e-9),
    }


def stage_resolutions(model: torch.nn.Module, device: torch.device) -> List[Dict[str, Any]]:
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
        result = call_model(model, torch.ones(1, 3, 224, 224, device=device))
    for handle in handles:
        handle.remove()

    features = None
    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, (tuple, list)) and len(item) >= 4 and all(torch.is_tensor(t) for t in item[:4]):
                features = list(item[:4])
                break
    if features is None and len(captured) >= 4:
        features = [captured[idx] for idx in range(4)]
    if features is None:
        raise RuntimeError("Could not determine stage resolutions.")
    return [
        {
            "stage": idx + 1,
            "shape": list(feature.shape),
            "resolution": list(feature.shape[-2:]) if feature.ndim == 4 else None,
            "channels": int(feature.shape[1]) if feature.ndim == 4 else int(feature.shape[-1]),
        }
        for idx, feature in enumerate(features[:4])
    ]


def time_throughput(model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    passes = 100
    attempted = [32, 16, 8, 4, 2, 1]
    last_error = None
    for batch in attempted:
        try:
            x = torch.randn(batch, 3, 224, 224, device=device)
            with torch.no_grad():
                for _ in range(3):
                    call_model(model, x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(passes):
                    call_model(model, x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            return {
                "batch_size": batch,
                "passes": passes,
                "elapsed_seconds": elapsed,
                "images_per_second": batch * passes / elapsed,
                "device": str(device),
            }
        except RuntimeError as exc:
            last_error = str(exc)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {"error": last_error or "throughput timing failed", "device": str(device)}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading nvidia/MambaVision-T-1K on {device}...")
    model = load_model().to(device).eval()
    mixer_name, mixer = first_mixer(model)

    x1 = torch.ones(1, 3, 224, 224, device=device)
    x2 = torch.rand(1, 3, 224, 224, device=device)
    hidden_1 = capture_input(model, mixer, x1)
    hidden_2 = capture_input(model, mixer, x2)
    params_1 = selective_parameters(mixer, hidden_1)
    params_2 = selective_parameters(mixer, hidden_2)
    delta_diff = float((params_1["delta"] - params_2["delta"]).abs().mean().cpu())
    b_diff = float((params_1["B"] - params_2["B"]).abs().mean().cpu())
    c_diff = float((params_1["C"] - params_2["C"]).abs().mean().cpu())

    ssm_future = future_token_access(mixer.conv1d_x, mixer.d_inner // 2)
    sym_future = future_token_access(mixer.conv1d_z, mixer.d_inner // 2)
    ssm_branch, sym_branch = branch_outputs(mixer, hidden_2)
    concat_width = ssm_branch.shape[1] + sym_branch.shape[1]
    fusion = {
        "mixer": mixer_name,
        "ssm_branch_channels": int(ssm_branch.shape[1]),
        "symmetric_branch_channels": int(sym_branch.shape[1]),
        "concatenated_channels": int(concat_width),
        "out_proj_in_features": int(mixer.out_proj.in_features),
        "verified_concat_before_projection": bool(concat_width == mixer.out_proj.in_features),
        "ssm_entropy_bits": entropy(ssm_branch),
        "symmetric_entropy_bits": entropy(sym_branch),
    }

    resolutions = stage_resolutions(model, device)
    throughput = time_throughput(model, device)

    results = {
        "selective_scan_input_dependence": {
            "mixer": mixer_name,
            "mean_abs_delta_difference_two_inputs": delta_diff,
            "mean_abs_B_difference_two_inputs": b_diff,
            "mean_abs_C_difference_two_inputs": c_diff,
            "verified": bool(delta_diff > 1e-8 and b_diff > 1e-8 and c_diff > 1e-8),
        },
        "symmetric_branch_non_causal_convolution": {
            "ssm_branch_conv1d_x": ssm_future,
            "symmetric_branch_conv1d_z": sym_future,
            "interpretation": "padding='same' depthwise Conv1d lets output token t depend on token t+1, so this is regular/non-causal local mixing.",
        },
        "concatenation_fusion": fusion,
        "stage_resolution_reduction": resolutions,
        "throughput": throughput,
        "claimed_pareto_front_note": "Compare measured images/s with the throughput table in the MambaVision paper for matching hardware and precision.",
    }

    print("\nMathematical verification summary")
    print("=" * 96)
    print(f"Selective scan input dependence: delta diff={delta_diff:.6g}, B diff={b_diff:.6g}, C diff={c_diff:.6g}")
    print(
        "Future-token access: "
        f"SSM conv={ssm_future['future_access_detected']}, "
        f"symmetric conv={sym_future['future_access_detected']}"
    )
    print(
        f"Concatenation fusion: {fusion['ssm_branch_channels']} + {fusion['symmetric_branch_channels']} "
        f"= {fusion['concatenated_channels']} -> out_proj.in_features={fusion['out_proj_in_features']}"
    )
    print("Stage resolutions for 224x224 input:")
    for item in resolutions:
        print(f"  Stage {item['stage']}: shape={item['shape']}")
    if "images_per_second" in throughput:
        print(
            f"Throughput: {throughput['images_per_second']:.2f} images/s "
            f"(batch={throughput['batch_size']}, passes={throughput['passes']}, device={throughput['device']})"
        )
    else:
        print(f"Throughput timing failed: {throughput.get('error')}")

    json_path = JSON_DIR / "mathematical_verification.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    (REPORT_DIR / "07_mathematical_verification.md").write_text(
        "# Mathematical Verification\n\n"
        f"- Structured results: `{json_path.relative_to(ROOT)}`\n"
        f"- Selective scan input-dependent delta difference: `{delta_diff:.6g}`\n"
        f"- Concat fusion verified: `{fusion['verified_concat_before_projection']}`\n",
        encoding="utf-8",
    )
    print(f"\nSaved mathematical verification JSON to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
