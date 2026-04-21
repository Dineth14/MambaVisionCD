#!/usr/bin/env python3
"""
SSM eigenspectrum, impulse response, frequency response, and delta analysis.

MambaVisionMixer stores A as log-positive values and uses A = -exp(A_log).
For a representative delta, the discrete transition is A_bar = exp(delta A).
Because A is diagonal in this implementation, stability reduces to checking
that |exp(delta * A_i)| < 1 for positive delta.
"""

from __future__ import annotations

import json
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


def ssm_layers(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    return [(name, module) for name, module in model.named_modules() if module.__class__.__name__ == "MambaVisionMixer"]


def learned_delta(module: torch.nn.Module) -> torch.Tensor:
    """Use the learned dt bias as the baseline delta distribution."""
    bias = module.dt_proj.bias.detach().float()
    return F.softplus(bias)


def discrete_eigenvalues(module: torch.nn.Module) -> torch.Tensor:
    a_cont = -torch.exp(module.A_log.detach().float())
    delta = learned_delta(module).mean()
    return torch.exp(delta * a_cont)


def impulse_response(module: torch.nn.Module, steps: int = 257) -> torch.Tensor:
    """Approximate channel impulse response using diagonal A_bar and unit B/C."""
    eig = discrete_eigenvalues(module).clamp(min=0)
    n = torch.arange(steps, dtype=eig.dtype, device=eig.device)
    return eig.unsqueeze(-1) ** n.unsqueeze(0)


def half_life_samples(response: torch.Tensor) -> float:
    # response has shape [channels, d_state, samples] for MambaVision's
    # diagonal A matrix. Average all state/channel axes before finding the
    # sample where the impulse drops below half its initial amplitude.
    mean_response = response.reshape(-1, response.shape[-1]).mean(dim=0)
    below = torch.nonzero(mean_response <= 0.5, as_tuple=False)
    if len(below) == 0:
        return float(response.shape[-1] - 1)
    return float(below[0].item())


def dominant_frequency(response: torch.Tensor) -> float:
    spectrum = torch.fft.rfft(response.float().reshape(-1, response.shape[-1]), dim=-1).abs().mean(dim=0)
    idx = int(torch.argmax(spectrum[1:]).item() + 1) if spectrum.numel() > 1 else 0
    return float(idx / max(1, 2 * (spectrum.numel() - 1)))


def plot_eigenspectrum(records: List[Dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    circle = plt.Circle((0, 0), 1.0, fill=False, color="#6b7f94", linestyle="--", linewidth=1.2)
    ax.add_patch(circle)
    for record in records:
        eig = np.asarray(record["eigenvalues"])
        ax.scatter(eig.real, eig.imag, s=10, alpha=0.55, label=record["name"].split(".")[-3:])
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("SSM Discrete Eigenspectrum")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    out = FIG_DIR / "ssm_eigenspectrum.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_impulse(records: List[Dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for record in records:
        response = np.asarray(record["impulse_response_mean"])
        ax.plot(response, label=record["short_name"])
    ax.set_xlabel("sample n")
    ax.set_ylabel("mean h[n]")
    ax.set_title("SSM Impulse Response Decay")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    out = FIG_DIR / "ssm_impulse_response.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_frequency(records: List[Dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for record in records:
        response = np.asarray(record["impulse_response_mean"])
        spectrum = np.abs(np.fft.rfft(response))
        freqs = np.fft.rfftfreq(len(response), d=1.0)
        ax.plot(freqs, spectrum, label=record["short_name"])
    ax.set_xlabel("normalized frequency omega / 2pi")
    ax.set_ylabel("|DFT(h[n])|")
    ax.set_title("SSM Frequency Response from Impulse Response")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    out = FIG_DIR / "ssm_frequency_response.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_delta(all_delta: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(all_delta, bins=60, color="#00d4ff", alpha=0.78)
    ax.axvline(np.mean(all_delta), color="#39ff8f", linestyle="--", label=f"mean={np.mean(all_delta):.4f}")
    ax.set_xlabel("learned delta/timescale")
    ax.set_ylabel("count")
    ax.set_title("Distribution of Learned Delta Values")
    ax.text(
        0.98,
        0.88,
        "small delta: slower state update, longer memory\nlarge delta: faster update, shorter memory",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#e8edf3",
        fontsize=9,
        bbox=dict(facecolor="#0d1620", edgecolor="#6b7f94", alpha=0.8),
    )
    ax.legend()
    out = FIG_DIR / "ssm_delta_distribution.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading nvidia/MambaVision-T-1K on {device}...")
    model = load_model().to(device).eval()
    layers = ssm_layers(model)
    if not layers:
        raise RuntimeError("No MambaVisionMixer SSM layers found.")

    records: List[Dict[str, Any]] = []
    all_delta = []
    for idx, (name, module) in enumerate(layers):
        eig = discrete_eigenvalues(module).cpu()
        response = impulse_response(module).cpu()
        delta = learned_delta(module).cpu().numpy()
        all_delta.append(delta)
        records.append(
            {
                "name": name,
                "short_name": f"SSM {idx}",
                "d_state": int(getattr(module, "d_state", module.A_log.shape[-1])),
                "d_model": int(getattr(module, "d_model", module.A_log.shape[0])),
                "mean_eigenvalue_magnitude": float(eig.abs().mean()),
                "max_eigenvalue_magnitude": float(eig.abs().max()),
                "stable_inside_unit_circle": bool((eig.abs() < 1.0).all()),
                "mean_impulse_response_half_life_samples": half_life_samples(response),
                "dominant_frequency_response_peak": dominant_frequency(response),
                "delta_mean": float(delta.mean()),
                "delta_min": float(delta.min()),
                "delta_max": float(delta.max()),
                "eigenvalues": eig.numpy().astype(complex).tolist(),
                "impulse_response_mean": response.reshape(-1, response.shape[-1]).mean(dim=0).numpy().tolist(),
            }
        )

    eig_path = plot_eigenspectrum(records)
    impulse_path = plot_impulse(records)
    freq_path = plot_frequency(records)
    delta_path = plot_delta(np.concatenate(all_delta))

    print("\nSSM mathematical summary")
    print("=" * 118)
    print(f"{'block':38s} {'d_state':>8s} {'d_model':>8s} {'mean|eig|':>11s} {'half-life':>10s} {'peak omega':>11s} stable")
    for record in records:
        print(
            f"{record['name'][:38]:38s} {record['d_state']:8d} {record['d_model']:8d} "
            f"{record['mean_eigenvalue_magnitude']:11.5f} {record['mean_impulse_response_half_life_samples']:10.1f} "
            f"{record['dominant_frequency_response_peak']:11.4f} {record['stable_inside_unit_circle']}"
        )

    json_path = JSON_DIR / "ssm_eigenspectrum_analysis.json"
    with json_path.open("w", encoding="utf-8") as f:
        # Convert complex eigenvalues into explicit real/imag records for JSON.
        serializable = []
        for record in records:
            item = dict(record)
            eig_np = np.asarray(item.pop("eigenvalues")).reshape(-1)
            item["eigenvalues"] = [{"real": float(np.real(v)), "imag": float(np.imag(v))} for v in eig_np]
            serializable.append(item)
        json.dump(
            {
                "model_id": "nvidia/MambaVision-T-1K",
                "records": serializable,
                "figures": [
                    str(eig_path.relative_to(ROOT)),
                    str(impulse_path.relative_to(ROOT)),
                    str(freq_path.relative_to(ROOT)),
                    str(delta_path.relative_to(ROOT)),
                ],
            },
            f,
            indent=2,
        )
    (REPORT_DIR / "06_ssm_eigenspectrum.md").write_text(
        "# SSM Eigenspectrum Analysis\n\n"
        f"- Eigenspectrum: `{eig_path.relative_to(ROOT)}`\n"
        f"- Impulse response: `{impulse_path.relative_to(ROOT)}`\n"
        f"- Frequency response: `{freq_path.relative_to(ROOT)}`\n"
        f"- Delta distribution: `{delta_path.relative_to(ROOT)}`\n",
        encoding="utf-8",
    )
    print(f"\nSaved SSM analysis JSON to {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
