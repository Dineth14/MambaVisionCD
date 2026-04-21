#!/usr/bin/env python3
"""
Spatial frequency and phase response probing for MambaVision-T-1K.

Sinusoidal gratings are eigenfunctions of linear shift-invariant filters. A
deep vision model is nonlinear and not shift-invariant, but gratings still
provide a controlled probe of which spatial frequencies and orientations
produce high activation magnitude in each hierarchical stage.
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


plt.style.use("dark_background")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "analysis" / "results" / "figures"
JSON_DIR = ROOT / "analysis" / "results" / "json"
REPORT_DIR = ROOT / "analysis" / "results" / "reports"
for directory in (FIG_DIR, JSON_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FREQUENCIES = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50], dtype=np.float32)
ORIENTATIONS_DEG = np.array([0, 45, 90, 135], dtype=np.float32)


def load_model() -> torch.nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)


def call_model(model: torch.nn.Module, x: torch.Tensor) -> Any:
    try:
        return model(x)
    except TypeError:
        return model(pixel_values=x)


def find_stage_modules(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    if hasattr(model, "levels"):
        return [(f"levels.{idx}", model.levels[idx]) for idx in range(min(4, len(model.levels)))]
    for name, module in model.named_modules():
        if name.endswith("levels") and hasattr(module, "__len__"):
            try:
                return [(f"{name}.{idx}", module[idx]) for idx in range(min(4, len(module)))]
            except Exception:
                continue
    return []


def output_features_from_result(result: Any) -> List[torch.Tensor]:
    """Prefer the HF feature tuple; otherwise return an empty list for hook fallback."""
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


def stage_features(model: torch.nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """Extract four stage feature maps with hooks if the forward result omits them."""
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    for idx, (_name, module) in enumerate(find_stage_modules(model)[:4], start=0):
        def make_hook(stage_idx: int):
            def _hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                candidate = output[0] if isinstance(output, (tuple, list)) else output
                if torch.is_tensor(candidate):
                    captured[stage_idx] = candidate.detach()
            return _hook
        handles.append(module.register_forward_hook(make_hook(idx)))
    with torch.no_grad():
        result = call_model(model, x)
    for handle in handles:
        handle.remove()
    features = output_features_from_result(result)
    if len(features) >= 4:
        return [feature.detach() for feature in features[:4]]
    if len(captured) >= 4:
        return [captured[idx] for idx in range(4)]
    raise RuntimeError("Could not extract four stage outputs.")


def grating(frequency: float, theta_deg: float, size: int = 224) -> torch.Tensor:
    """Generate a 3-channel sine grating in cycles/pixel at orientation theta."""
    theta = math.radians(float(theta_deg))
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    phase_coordinate = x * math.cos(theta) + y * math.sin(theta)
    image = 0.5 + 0.5 * np.sin(2.0 * math.pi * frequency * phase_coordinate)
    image = np.stack([image, image, image], axis=0)
    return torch.from_numpy(image).float()


def feature_magnitude(feature: torch.Tensor) -> float:
    return float(feature.detach().abs().mean().cpu())


def plot_frequency_tuning(response: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for stage in range(4):
        ax = axes[stage]
        for ori_idx, theta in enumerate(ORIENTATIONS_DEG):
            ax.plot(FREQUENCIES, response[stage, :, ori_idx], marker="o", label=f"{int(theta)} deg")
        ax.set_title(f"Stage {stage + 1}")
        ax.set_xlabel("cycles/pixel")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("mean |activation|")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Frequency Tuning per Stage")
    out = FIG_DIR / "frequency_tuning_per_stage.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_orientation_polar(response: np.ndarray) -> Path:
    fig, axes = plt.subplots(1, 4, subplot_kw={"projection": "polar"}, figsize=(18, 4.5))
    angles = np.deg2rad(np.r_[ORIENTATIONS_DEG, ORIENTATIONS_DEG[0]])
    for stage in range(4):
        values = response[stage].mean(axis=0)
        values = np.r_[values, values[0]]
        axes[stage].plot(angles, values, color="#00d4ff", marker="o")
        axes[stage].fill(angles, values, color="#00d4ff", alpha=0.18)
        axes[stage].set_title(f"Stage {stage + 1}")
    fig.suptitle("Orientation Tuning Averaged over Frequency")
    out = FIG_DIR / "orientation_tuning_polar.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_frequency_heatmap(response: np.ndarray) -> Path:
    heat = response.mean(axis=2)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    im = ax.imshow(heat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(4), [f"Stage {idx}" for idx in range(1, 5)])
    ax.set_xticks(range(len(FREQUENCIES)), [f"{f:.2f}" for f in FREQUENCIES])
    ax.set_xlabel("cycles/pixel")
    ax.set_title("Stage x Frequency Activation Heatmap")
    fig.colorbar(im, ax=ax, label="mean |activation|")
    out = FIG_DIR / "frequency_heatmap.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def radial_power_spectrum(feature: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, float]:
    fmap = feature.detach().float()[0].mean(dim=0).cpu().numpy()
    fmap = fmap - fmap.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(fmap))) ** 2
    h, w = power.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2).astype(np.int32)
    max_r = rr.max()
    radial = np.array([power[rr == r].mean() if np.any(rr == r) else 0 for r in range(max_r + 1)])
    freqs = np.arange(max_r + 1) / max(h, w)
    peak_idx = int(np.argmax(radial[1:]) + 1) if len(radial) > 1 else 0
    return freqs, radial, float(freqs[peak_idx])


def plot_power_spectrum(model: torch.nn.Module, device: torch.device) -> Tuple[Path, List[float]]:
    # Uniform random noise is a controlled ImageNet-style broad-band stimulus.
    x = torch.rand(1, 3, 224, 224, device=device)
    features = stage_features(model, x)
    fig, ax = plt.subplots(figsize=(9, 5))
    peaks = []
    for stage, feature in enumerate(features, start=1):
        freqs, radial, peak = radial_power_spectrum(feature)
        peaks.append(peak)
        ax.plot(freqs, np.log10(radial + 1e-12), label=f"Stage {stage}, peak={peak:.3f}")
        ax.axvline(peak, color="#6b7f94", lw=0.7, alpha=0.35)
    ax.set_xlabel("normalized radial frequency")
    ax.set_ylabel("log10 power")
    ax.set_title("Radially Averaged Feature Map Power Spectrum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.22)
    out = FIG_DIR / "feature_map_power_spectrum.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out, peaks


def dominant_phase(feature: torch.Tensor) -> float:
    fmap = feature.detach().float()[0].mean(dim=0).cpu().numpy()
    spectrum = np.fft.fft2(fmap - fmap.mean())
    mag = np.abs(spectrum)
    mag[0, 0] = 0
    idx = np.unravel_index(np.argmax(mag), mag.shape)
    return float(np.angle(spectrum[idx]))


def phase_response(model: torch.nn.Module, device: torch.device) -> np.ndarray:
    phase = np.zeros((4, len(FREQUENCIES)), dtype=np.float32)
    for f_idx, freq in enumerate(FREQUENCIES):
        stage_phase_values = []
        for theta in ORIENTATIONS_DEG:
            x = grating(float(freq), float(theta)).unsqueeze(0).to(device)
            features = stage_features(model, x)
            stage_phase_values.append([dominant_phase(feature) for feature in features])
        phase[:, f_idx] = np.unwrap(np.array(stage_phase_values), axis=0).mean(axis=0)
    return phase


def plot_phase_response(phase: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for stage in range(4):
        ax.plot(FREQUENCIES, phase[stage], marker="o", label=f"Stage {stage + 1}")
    ax.set_xlabel("cycles/pixel")
    ax.set_ylabel("dominant output phase (radians)")
    ax.set_title("Phase Response per Stage")
    ax.grid(alpha=0.22)
    ax.legend()
    out = FIG_DIR / "phase_response_per_stage.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading nvidia/MambaVision-T-1K on {device}...")
    model = load_model().to(device).eval()

    response = np.zeros((4, len(FREQUENCIES), len(ORIENTATIONS_DEG)), dtype=np.float32)
    for f_idx, freq in enumerate(FREQUENCIES):
        for o_idx, theta in enumerate(ORIENTATIONS_DEG):
            print(f"Stimulus frequency={freq:.2f} cycles/pixel, orientation={int(theta)} deg")
            x = grating(float(freq), float(theta)).unsqueeze(0).to(device)
            features = stage_features(model, x)
            response[:, f_idx, o_idx] = [feature_magnitude(feature) for feature in features]

    tuning_path = plot_frequency_tuning(response)
    polar_path = plot_orientation_polar(response)
    heatmap_path = plot_frequency_heatmap(response)
    spectrum_path, peaks = plot_power_spectrum(model, device)
    phase = phase_response(model, device)
    phase_path = plot_phase_response(phase)

    dominant_stage_by_freq = response.mean(axis=2).argmax(axis=0) + 1
    print("\nFrequency response summary")
    print("=" * 80)
    for freq, stage in zip(FREQUENCIES, dominant_stage_by_freq):
        print(f"{freq:.2f} cycles/pixel -> strongest mean activation in Stage {int(stage)}")
    for idx, peak in enumerate(peaks, start=1):
        print(f"Stage {idx} feature-map dominant radial frequency peak: {peak:.4f}")

    json_path = JSON_DIR / "frequency_phase_response.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "frequencies_cycles_per_pixel": FREQUENCIES.tolist(),
                "orientations_degrees": ORIENTATIONS_DEG.tolist(),
                "response_shape": list(response.shape),
                "frequency_response": response.tolist(),
                "phase_response_radians": phase.tolist(),
                "dominant_stage_by_frequency": dominant_stage_by_freq.tolist(),
                "power_spectrum_peak_by_stage": peaks,
                "figures": [
                    str(tuning_path.relative_to(ROOT)),
                    str(polar_path.relative_to(ROOT)),
                    str(heatmap_path.relative_to(ROOT)),
                    str(spectrum_path.relative_to(ROOT)),
                    str(phase_path.relative_to(ROOT)),
                ],
            },
            f,
            indent=2,
        )

    (REPORT_DIR / "03_frequency_phase_response.md").write_text(
        "# Frequency and Phase Response\n\n"
        f"- Frequency tuning: `{tuning_path.relative_to(ROOT)}`\n"
        f"- Orientation polar plots: `{polar_path.relative_to(ROOT)}`\n"
        f"- Heatmap: `{heatmap_path.relative_to(ROOT)}`\n"
        f"- Power spectrum: `{spectrum_path.relative_to(ROOT)}`\n"
        f"- Phase response: `{phase_path.relative_to(ROOT)}`\n",
        encoding="utf-8",
    )
    print(f"\nSaved JSON results to {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
