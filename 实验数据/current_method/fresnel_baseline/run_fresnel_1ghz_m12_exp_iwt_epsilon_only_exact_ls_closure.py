from __future__ import annotations

"""Epsilon-only exact-LS synthetic closure at 1 GHz.

There is no trainable internal-field branch.  Each Adam step differentiates
through a 32x32 multi-RHS Lippmann-Schwinger solve using the M=12 provider.
"""

import csv
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_fresnel_1ghz_m12_exp_iwt_synthetic_closure as closure
from square_target import pinn_pixel_inverse_core as core


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "results_fresnel_1GHz_m12_exp_iwt_epsilon_only_exact_ls_closure1000"
SEED = 20260722
GRID_SIZE = 32
STEPS = 1000
SAVE_EVERY = 100
EPS = 1.0e-12


class EpsilonOnlyLS(torch.nn.Module):
    def __init__(self, config: core.TrainConfig, target: core.TargetSpec) -> None:
        super().__init__()
        self.epsilon_branch = core.EpsilonBranch(config, target)
        final_layer = self.epsilon_branch.mlp.net[-1]
        if not isinstance(final_layer, torch.nn.Linear):
            raise TypeError("Expected a Linear epsilon final layer.")
        # Preserve mean epsilon from the constant-init bias while avoiding an
        # exactly spatially symmetric first receiver-loss gradient.
        torch.nn.init.normal_(final_layer.weight, mean=0.0, std=1.0e-3)

    def epsilon(self, xy: torch.Tensor) -> torch.Tensor:
        return self.epsilon_branch(xy)


def exact_ls_receiver_prediction(model: EpsilonOnlyLS, direct_ls: Any, incident: closure.M12IncidentProvider, config: core.TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate Es_receiver through the shared 36-RHS LS system."""
    integral = direct_ls.integral
    epsilon = model.epsilon(integral.quad_xy).reshape(-1, 1)
    chi = (epsilon - 1.0).to(direct_ls.domain_green.dtype)
    scale = float(config.k0**2 * integral.area_weight)
    n_cells = epsilon.numel()
    system = torch.eye(n_cells, dtype=direct_ls.domain_green.dtype, device=epsilon.device) - scale * direct_ls.domain_green * chi.reshape(1, -1)
    total_field = torch.linalg.solve(system, incident.quad)
    source = chi * total_field
    receiver_green = torch.complex(integral.green_re, integral.green_im).to(direct_ls.domain_green.dtype)
    return scale * (receiver_green @ source), epsilon


def normalized_receiver_loss(prediction: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(prediction - observation).square()) / (torch.sum(torch.abs(observation).square()) + EPS)


def epsilon_image(model: EpsilonOnlyLS, target: core.TargetSpec, device: torch.device, dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-target.roi_half_width, target.roi_half_width, 256, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    xy = torch.as_tensor(np.column_stack((xx.reshape(-1), yy.reshape(-1))), dtype=dtype, device=device)
    with torch.no_grad():
        epsilon = model.epsilon(xy).reshape(xx.shape).cpu().numpy()
    return xx, yy, epsilon


def save_epsilon(epsilon: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(epsilon, extent=[-0.09, 0.09, -0.09, 0.09], origin="lower", cmap="jet", vmin=1.0, vmax=4.0, interpolation="bilinear")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Relative Permittivity")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def offline_metrics(epsilon: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> dict[str, float]:
    truth = np.where((xx - closure.SYNTHETIC_CENTER[0]) ** 2 + (yy - closure.SYNTHETIC_CENTER[1]) ** 2 <= closure.SYNTHETIC_RADIUS**2, closure.SYNTHETIC_EPSILON, 1.0)
    mask = truth > 1.0
    return {
        "continuous_RE": float(np.linalg.norm(epsilon - truth) / np.linalg.norm(truth)),
        "continuous_SSIM": core.global_ssim(epsilon, truth, closure.SYNTHETIC_EPSILON - 1.0),
        "target_mean": float(epsilon[mask].mean()),
        "target_max": float(epsilon[mask].max()),
        "background_mean": float(epsilon[~mask].mean()),
        "background_std": float(epsilon[~mask].std()),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {OUTPUT_DIR}")
    config = closure.make_config()
    config.integral_grid_size = GRID_SIZE
    config.learning_rate = 1.0e-3
    target = core.TargetSpec(name="Fresnel epsilon-only synthetic closure", kind="circle", eps_background=1.0, eps_object=4.0, roi_half_width=0.09)
    device, dtype = core.resolve_device(config.device), core.resolve_dtype(config.dtype)
    measurement, _indices, _directions, fits, fit_metadata = closure.make_geometry_and_m12(config)

    class Geometry:
        xy = closure.base.mapped_receiver_xy(measurement)

    integral = core.make_integral_tensors(Geometry(), target, config, device, dtype)
    if integral is None:
        raise RuntimeError("Exact-LS closure requires static integral tensors.")
    direct_ls = core.make_direct_ls_tensors(integral, config, device, dtype)
    closure.base.replace_with_exp_iwt_green(direct_ls, Geometry.xy, config.k0)
    incident = closure.M12IncidentProvider(fits, integral.quad_xy, device, direct_ls.domain_green.dtype)
    synthetic_observation = closure.synthesize_receiver_data(direct_ls, incident, config)
    core.set_global_seed(SEED)
    model = EpsilonOnlyLS(config, target).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    image_dir = OUTPUT_DIR / "epsilon_images"; image_dir.mkdir()
    np.save(OUTPUT_DIR / "synthetic_receiver_scattered_m12.npy", synthetic_observation.detach().cpu().numpy())
    (OUTPUT_DIR / "config.json").write_text(json.dumps({
        "config": asdict(config),
        "convention": {"time": "exp(+i omega t)", "incident": "M12IncidentProvider only", "green": "-i/4 H0^(2)(k0*distance)", "chi": "epsilon-1"},
        "forward": "(I-k0^2*G_D*diag(chi)*dA) Etotal=Einc; Es_receiver=k0^2*G_R*(chi*Etotal)*dA",
        "loss": "normalized receiver loss only; no trainable Es_inside, PDE, boundary, TV, material, or geometry prior",
        "epsilon_initialization": {"mean": 1.5, "final_weight_std": 1.0e-3, "bias": "constant logit for eps_initial"},
        "synthetic_generation_only": {"circle_center_m": list(closure.SYNTHETIC_CENTER), "radius_m": closure.SYNTHETIC_RADIUS, "epsilon": closure.SYNTHETIC_EPSILON, "not_used_for_training_or_selection": True},
        "incident_fit_metadata": fit_metadata,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    started = time.time()
    for step in range(1, STEPS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction, _epsilon_grid = exact_ls_receiver_prediction(model, direct_ls, incident, config)
        receiver_loss = normalized_receiver_loss(prediction, synthetic_observation)
        if not bool(torch.isfinite(receiver_loss)):
            raise FloatingPointError(f"Non-finite exact-LS receiver loss at step {step}.")
        receiver_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm); optimizer.step()
        if step % SAVE_EVERY:
            continue
        model.eval()
        with torch.no_grad():
            full_prediction, _epsilon_grid = exact_ls_receiver_prediction(model, direct_ls, incident, config)
            full_loss = normalized_receiver_loss(full_prediction, synthetic_observation)
            xx, yy, epsilon = epsilon_image(model, target, device, dtype)
        loss_value = float(full_loss.cpu())
        offline = offline_metrics(epsilon, xx, yy)
        save_epsilon(epsilon, image_dir / f"epsilon_step_{step:04d}.png", f"Epsilon-only exact LS closure, step {step}")
        checkpoint = OUTPUT_DIR / f"checkpoint_step_{step:04d}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": asdict(config)}, checkpoint)
        if loss_value < best_loss:
            best_loss = loss_value; shutil.copy2(checkpoint, OUTPUT_DIR / "best_checkpoint.pt")
        row = {"step": step, "L_receiver": loss_value, "epsilon_min": float(epsilon.min()), "epsilon_max": float(epsilon.max()), "epsilon_mean": float(epsilon.mean()), "epsilon_std": float(epsilon.std()), "elapsed_s": time.time() - started, **offline}
        history.append(row); write_csv(history, OUTPUT_DIR / "history.csv")
        print(f"step={step:4d} Lreceiver={loss_value:.4e} epsmax={row['epsilon_max']:.3f} target_mean={offline['target_mean']:.3f} RE/SSIM={offline['continuous_RE']:.4f}/{offline['continuous_SSIM']:.4f}", flush=True)

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": STEPS, "config": asdict(config)}, OUTPUT_DIR / "final_checkpoint.pt")
    selected = torch.load(OUTPUT_DIR / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"]); model.eval()
    xx, yy, epsilon = epsilon_image(model, target, device, dtype)
    save_epsilon(epsilon, OUTPUT_DIR / "best_continuous_reconstruction.png", f"Epsilon-only exact LS best, step {selected['step']}")
    summary = {"steps_completed": STEPS, "best_checkpoint_step": int(selected["step"]), "best_receiver_loss": best_loss, "best_epsilon": {"min": float(epsilon.min()), "max": float(epsilon.max()), "mean": float(epsilon.mean()), "std": float(epsilon.std())}, "offline_synthetic_evaluation_only": offline_metrics(epsilon, xx, yy), "elapsed_s": time.time() - started, "real_fresnel_training_started": False}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
