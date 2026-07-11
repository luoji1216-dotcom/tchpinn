from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinn_pixel_inverse_core import (
    DoubleBranchPINN,
    TargetSpec,
    TrainConfig,
    compute_loss_terms,
    load_observations,
    make_integral_tensors,
    resolve_device,
    resolve_dtype,
    sample_boundary_points,
    sample_collocation_points,
    sample_direction_batch,
    sample_directions,
    sample_observation_batch,
    set_global_seed,
    to_observation_tensors,
)


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().norm().cpu()) ** 2
    return math.sqrt(total)


def load_train_config(config_path: Path, preset: str) -> TrainConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = payload["config"]
    return TrainConfig(
        frequency_hz=cfg["frequency_hz"],
        incident_amplitude=cfg["incident_amplitude"],
        incident_phase_sign=cfg["incident_phase_sign"],
        observation_imag_sign=cfg["observation_imag_sign"],
        estimate_incident_amplitude=cfg["estimate_incident_amplitude"],
        eps_min=cfg["eps_min"],
        eps_max=cfg["eps_max"],
        eps_initial=cfg["eps_initial"],
        domain_radius=cfg["domain_radius"],
        max_points_per_direction=cfg["max_points_per_direction"],
        data_batch_per_direction=cfg["data_batch_per_direction"],
        n_pde=cfg["n_pde"],
        n_boundary=cfg["n_boundary"],
        n_tv_grid=cfg["n_tv_grid"],
        integral_grid_size=cfg["integral_grid_size"],
        plot_grid_size=cfg["plot_grid_size"],
        epochs_adam=0,
        epochs_lbfgs=0,
        learning_rate=cfg["learning_rate"],
        weight_data=cfg["weight_data"],
        weight_pde=cfg["weight_pde"],
        weight_boundary=cfg["weight_boundary"],
        weight_integral_data=cfg["weight_integral_data"],
        weight_tv=cfg["weight_tv"],
        weight_contrast_l1=cfg["weight_contrast_l1"],
        robust_data_weighting=cfg["robust_data_weighting"],
        loss_preset=preset,
        lambda_f=cfg.get("lambda_f", 1.0),
        lambda_d=cfg.get("lambda_d", 100.0),
        lambda_ep=cfg.get("lambda_ep", 100.0),
        adaptive_gamma=cfg.get("adaptive_gamma", 1.0),
        adaptive_delta=cfg.get("adaptive_delta", 1.0e-8),
        edge_delta=cfg.get("edge_delta", 1.0e-3),
        field_hidden_layers=cfg["field_hidden_layers"],
        field_hidden_units=cfg["field_hidden_units"],
        eps_hidden_layers=cfg["eps_hidden_layers"],
        eps_hidden_units=cfg["eps_hidden_units"],
        fourier_bands=cfg["fourier_bands"],
        fourier_max_frequency=cfg["fourier_max_frequency"],
        random_seed=cfg["random_seed"],
        dtype=cfg["dtype"],
        device=cfg["device"],
        log_every=cfg["log_every"],
        checkpoint_every=cfg["checkpoint_every"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="square_target/results_5_3_2_square_0_3GHz_full_paper_30000/run_config.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="square_target/results_5_3_2_square_0_3GHz_full_paper_30000/checkpoint_adam_030000.pt",
    )
    args = parser.parse_args()

    root = Path.cwd()
    target = TargetSpec(
        name="Regular square target",
        kind="square",
        eps_background=1.0,
        eps_object=4.0,
        roi_half_width=0.5,
        square_side=0.4,
    )
    config = load_train_config(root / args.base_config, "paper_plus_integral")
    set_global_seed(config.random_seed)
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)

    obs = load_observations(root / "square_target", config, direction_labels=("+x", "-x"))
    obs_tensors = to_observation_tensors(obs, device=device, dtype=dtype)
    integral_tensors = make_integral_tensors(obs, target, config, device=device, dtype=dtype)
    radius = config.domain_radius or obs.receiver_radius * 0.98

    data_indices, data_xy, data_dirs, _data_amps, data_target = sample_observation_batch(
        obs_tensors, config.data_batch_per_direction
    )
    pde_xy = sample_collocation_points(config.n_pde, target, radius, device, dtype)
    pde_dirs, pde_amps = sample_direction_batch(
        obs_tensors.unique_directions, obs_tensors.unique_amplitudes, config.n_pde
    )
    bc_xy = sample_boundary_points(config.n_boundary, obs.receiver_radius, device, dtype)
    bc_dirs = sample_directions(obs_tensors.unique_directions, config.n_boundary)

    def make_losses(model: DoubleBranchPINN):
        return compute_loss_terms(
            model,
            data_xy=data_xy,
            data_dirs=data_dirs,
            data_target=data_target,
            data_indices=data_indices,
            pde_xy=pde_xy,
            pde_dirs=pde_dirs,
            pde_amps=pde_amps,
            bc_xy=bc_xy,
            bc_dirs=bc_dirs,
            integral_tensors=integral_tensors,
            obs_tensors=obs_tensors,
            target=target,
            config=config,
            device=device,
            dtype=dtype,
        )

    states: list[tuple[str, str | None]] = [
        ("init", None),
        ("full_paper_final", str(root / args.checkpoint)),
    ]
    components: list[tuple[str, Callable[[dict], torch.Tensor], Callable[[dict], torch.Tensor]]] = [
        ("Ld_w", lambda l: l["ldw"], lambda l: config.lambda_d * l["ldw"]),
        ("Lf", lambda l: l["lf"], lambda l: config.lambda_f * l["lf"]),
        ("Lep", lambda l: l["lep"], lambda l: config.lambda_ep * l["lep"]),
        ("integral_data", lambda l: l["integral_data"], lambda l: config.weight_integral_data * l["integral_data"]),
        (
            "total_paper",
            lambda l: config.lambda_f * l["lf"] + config.lambda_d * l["ldw"] + config.lambda_ep * l["lep"],
            lambda l: config.lambda_f * l["lf"] + config.lambda_d * l["ldw"] + config.lambda_ep * l["lep"],
        ),
        ("total_paper_plus_integral", lambda l: l["total"], lambda l: l["total"]),
    ]

    print(
        json.dumps(
            {
                "device": str(device),
                "dtype": str(dtype),
                "lambda_f": config.lambda_f,
                "lambda_d": config.lambda_d,
                "lambda_ep": config.lambda_ep,
                "weight_integral_data": config.weight_integral_data,
            },
            indent=2,
        )
    )

    for label, checkpoint in states:
        model = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            model.load_state_dict(state)
        model.train()

        with torch.no_grad():
            xs = torch.linspace(-target.roi_half_width, target.roi_half_width, 128, device=device, dtype=dtype)
            ys = torch.linspace(-target.roi_half_width, target.roi_half_width, 128, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            eps = model.epsilon(torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1))
            eps_summary = {
                "min": float(eps.min().cpu()),
                "max": float(eps.max().cpu()),
                "mean": float(eps.mean().cpu()),
            }

        print(f"\nSTATE {label}")
        print("eps", json.dumps(eps_summary))
        base_losses = make_losses(model)
        raw = {name: float(raw_fn(base_losses).detach().cpu()) for name, raw_fn, _ in components}
        weighted = {name: float(weighted_fn(base_losses).detach().cpu()) for name, _, weighted_fn in components}
        print("raw", json.dumps(raw, indent=2))
        print("weighted", json.dumps(weighted, indent=2))

        for name, _raw_fn, weighted_fn in components:
            model.zero_grad(set_to_none=True)
            losses = make_losses(model)
            loss = weighted_fn(losses)
            loss.backward()
            print(
                json.dumps(
                    {
                        "component": name,
                        "weighted_value": float(loss.detach().cpu()),
                        "epsilon_branch_grad_norm": grad_norm(model.epsilon_branch.parameters()),
                        "field_branch_grad_norm": grad_norm(model.field_branch.parameters()),
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
