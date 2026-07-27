"""Formal six-view Austria 0.3 GHz J-epsilon dual-branch inversion."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch

try:
    from .method import (
        GreenOperators,
        JEpsilonNetwork,
        TorchGreenOperators,
        back_projection,
        branch_parameters,
        load_method_config,
        normalized_terms,
        set_branch_trainable,
        tv_edge_loss,
    )
except ImportError:
    from method import (  # type: ignore[no-redef]
        GreenOperators,
        JEpsilonNetwork,
        TorchGreenOperators,
        back_projection,
        branch_parameters,
        load_method_config,
        normalized_terms,
        set_branch_trainable,
        tv_edge_loss,
    )


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_CONFIG = HERE / "formal_config.json"
DEFAULT_DATA = REPO_ROOT.parent / "data_austra" / "0.3GHz_six_direction"
DEFAULT_OUTPUT = HERE / "results_cst_j_epsilon_discrete_pde_inversion"
FORMAL_DIRECTIONS = np.asarray(
    (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
        (-1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)),
    ),
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def incident_fields(
    axis: np.ndarray,
    directions: np.ndarray,
    amplitude: float,
    phase_sign: float,
    k0: float,
) -> np.ndarray:
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    phase = (
        directions[:, 0, None, None] * xx[None]
        + directions[:, 1, None, None] * yy[None]
    )
    return amplitude * np.exp(1j * phase_sign * k0 * phase)


def load_formal_data(data_dir: Path, payload: dict, train_config: object):
    from pinn_pixel_inverse_core import load_observations

    labels = tuple(payload["direction_labels"])
    observations = load_observations(
        data_dir, train_config, direction_labels=labels
    )
    if tuple(observations.direction_labels) != labels:
        raise RuntimeError("formal direction order mismatch")
    measured_views = []
    xy_views = []
    directions = []
    for label in labels:
        indices = np.flatnonzero(observations.labels == label)
        measured_views.append(observations.scattered[indices])
        xy_views.append(observations.xy[indices])
        directions.append(observations.directions[indices[0]])
    counts = {values.shape[0] for values in measured_views}
    if len(counts) != 1:
        raise RuntimeError("all formal views must have equal receiver counts")
    receiver_xy = np.asarray(xy_views[0], dtype=np.float64)
    mismatch = max(
        float(np.max(np.linalg.norm(xy - receiver_xy, axis=1)))
        for xy in xy_views
    )
    if mismatch > float(payload["receiver_coordinate_tolerance_m"]):
        raise RuntimeError(f"cross-view receiver mismatch: {mismatch}")
    if not np.allclose(
        np.asarray(directions, dtype=np.float64),
        FORMAL_DIRECTIONS,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "formal six-direction vectors differ from the validated order"
        )
    if measured_views[0].shape[0] != 836:
        raise RuntimeError(
            f"formal CST smoke expects 836 points/view, got {counts}"
        )
    return (
        np.asarray(measured_views, dtype=np.complex128),
        receiver_xy,
        np.asarray(directions, dtype=np.float64),
    )


def predict(
    model: JEpsilonNetwork,
    xy_all: torch.Tensor,
    direction_all: torch.Tensor,
    xy_single: torch.Tensor,
    views: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    currents = model.current(xy_all, direction_all).reshape(views, n, n)
    epsilon = model.epsilon(xy_single).reshape(n, n)
    return currents, epsilon


def fit_bp_initialization(
    model: JEpsilonNetwork,
    xy_all: torch.Tensor,
    direction_all: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    tolerance: float,
    clip_norm: float,
) -> float:
    parameters = set_branch_trainable(model, "J")
    optimizer = torch.optim.Adam(parameters, lr=1.0e-3)
    error = math.inf
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.current(xy_all, direction_all)
        # Match the outer raw Re/Im mean-square loss exactly.  The complex
        # form has two components, so it carries a compensating one-half.
        loss = 0.5 * torch.mean(torch.abs(prediction - target).square())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        optimizer.step()
        with torch.no_grad():
            prediction = model.current(xy_all, direction_all)
            error = float(
                torch.linalg.vector_norm(prediction - target)
                / torch.linalg.vector_norm(target).clamp_min(1.0e-30)
            )
        if error < tolerance:
            break
    return error


def write_history(output_dir: Path, history: list[Dict[str, float]]) -> None:
    with (output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    payload, train_config = load_method_config(args.config)
    steps = payload["alternating_steps"] if args.steps is None else args.steps
    if payload["contrast_sparsity_weight"] != 0.0:
        raise ValueError("formal baseline requires zero contrast sparsity")
    np.random.seed(payload["seed"])
    torch.manual_seed(payload["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(payload["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        model = JEpsilonNetwork(train_config, payload["roi_half_width"]).to(device)
        print(
            json.dumps(
                {
                    "device": str(device),
                    "J_parameters": sum(
                        parameter.numel()
                        for parameter in branch_parameters(model, "J")
                    ),
                    "epsilon_parameters": sum(
                        parameter.numel()
                        for parameter in branch_parameters(model, "epsilon")
                    ),
                    "alternating_steps": steps,
                    "learning_rate_j": payload["learning_rate_j"],
                    "learning_rate_epsilon": payload["learning_rate_epsilon"],
                    "contrast_sparsity_weight": 0.0,
                }
            )
        )
        return

    measured_np, receiver_xy, directions_np = load_formal_data(
        args.data_dir, payload, train_config
    )
    axis = np.linspace(
        -payload["roi_half_width"],
        payload["roi_half_width"],
        payload["grid_size"],
        dtype=np.float64,
    )
    n = axis.size
    views = directions_np.shape[0]
    spacing = float(axis[1] - axis[0])
    k0 = 2.0 * math.pi * payload["frequency_hz"] / payload["c0"]
    incident_np = incident_fields(
        axis,
        directions_np,
        payload["incident_amplitude"],
        payload["incident_phase_sign"],
        k0,
    ).astype(np.complex128)
    operators_np = GreenOperators(axis, receiver_xy, k0)
    operators = TorchGreenOperators(operators_np, device)
    bp_np = back_projection(measured_np, operators_np)

    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    xy_np = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    xy_single = torch.as_tensor(xy_np, dtype=torch.float32, device=device)
    xy_all = xy_single.repeat(views, 1)
    direction_all = torch.as_tensor(
        np.repeat(directions_np, n * n, axis=0),
        dtype=torch.float32,
        device=device,
    )
    incident = torch.as_tensor(
        incident_np, dtype=torch.complex64, device=device
    )
    measured = torch.as_tensor(
        measured_np, dtype=torch.complex64, device=device
    )
    bp_target = torch.as_tensor(
        bp_np.reshape(-1), dtype=torch.complex64, device=device
    )

    model = JEpsilonNetwork(train_config, payload["roi_half_width"]).to(device)
    init_re = fit_bp_initialization(
        model,
        xy_all,
        direction_all,
        bp_target,
        payload["initial_j_fit_steps"],
        payload["initial_j_fit_tolerance"],
        payload["gradient_clip_norm"],
    )
    epsilon_final = model.epsilon_mlp.net[-1]
    initial_fraction = (
        payload["epsilon_initial"] - payload["epsilon_min"]
    ) / (payload["epsilon_max"] - payload["epsilon_min"])
    initial_bias = math.log(initial_fraction / (1.0 - initial_fraction))
    with torch.no_grad():
        epsilon_final.weight.zero_()
        epsilon_final.bias.fill_(initial_bias)

    j_optimizer = torch.optim.Adam(
        branch_parameters(model, "J"), lr=payload["learning_rate_j"]
    )
    epsilon_optimizer = torch.optim.Adam(
        branch_parameters(model, "epsilon"),
        lr=payload["learning_rate_epsilon"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[Dict[str, float]] = []

    for step in range(1, steps + 1):
        j_parameters = set_branch_trainable(model, "J")
        j_optimizer.zero_grad(set_to_none=True)
        currents, epsilon = predict(
            model, xy_all, direction_all, xy_single, views, n
        )
        j_terms = normalized_terms(
            currents,
            epsilon.detach(),
            incident,
            measured,
            operators,
            spacing,
            k0,
        )
        (
            payload["weight_data"] * j_terms["data_loss"]
            + payload["weight_state"] * j_terms["state_loss"]
            + payload["weight_pde"] * j_terms["pde_loss"]
        ).backward()
        torch.nn.utils.clip_grad_norm_(
            j_parameters, payload["gradient_clip_norm"]
        )
        j_optimizer.step()

        epsilon_parameters = set_branch_trainable(model, "epsilon")
        epsilon_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            currents_fixed = model.current(
                xy_all, direction_all
            ).reshape(views, n, n)
        epsilon = model.epsilon(xy_single).reshape(n, n)
        epsilon_terms = normalized_terms(
            currents_fixed.detach(),
            epsilon,
            incident,
            measured,
            operators,
            spacing,
            k0,
        )
        (
            payload["weight_state"] * epsilon_terms["state_loss"]
            + payload["weight_pde"] * epsilon_terms["pde_loss"]
            + payload["tv_edge_weight"] * tv_edge_loss(epsilon)
        ).backward()
        torch.nn.utils.clip_grad_norm_(
            epsilon_parameters, payload["gradient_clip_norm"]
        )
        epsilon_optimizer.step()

        if step % payload["log_every"] == 0 or step == steps:
            with torch.no_grad():
                currents, epsilon = predict(
                    model, xy_all, direction_all, xy_single, views, n
                )
                terms = normalized_terms(
                    currents,
                    epsilon,
                    incident,
                    measured,
                    operators,
                    spacing,
                    k0,
                )
                row = {
                    "step": step,
                    "data_loss": float(terms["data_loss"].cpu()),
                    "state_loss": float(terms["state_loss"].cpu()),
                    "pde_loss": float(terms["pde_loss"].cpu()),
                    "epsilon_min": float(epsilon.min().cpu()),
                    "epsilon_max": float(epsilon.max().cpu()),
                    "epsilon_mean": float(epsilon.mean().cpu()),
                }
            history.append(row)
            print(json.dumps(row), flush=True)
            write_history(args.output_dir, history)

        if step % payload["checkpoint_every"] == 0 or step == steps:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "J_optimizer": j_optimizer.state_dict(),
                    "epsilon_optimizer": epsilon_optimizer.state_dict(),
                    "BP_fit_RE": init_re,
                    "config": payload,
                    "history": history,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                },
                args.output_dir / f"checkpoint_step{step:05d}.pt",
            )


if __name__ == "__main__":
    main()
