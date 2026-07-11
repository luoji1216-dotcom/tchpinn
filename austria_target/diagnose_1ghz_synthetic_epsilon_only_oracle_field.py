from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.special import hankel1

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "square_target"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from austria_optimized_common import build_target  # noqa: E402
from pinn_pixel_inverse_core import (  # noqa: E402
    TrainConfig,
    epsilon_metrics,
    incident_field_numpy,
    load_fem_table,
    parse_direction_from_name,
    plot_comparison,
    target_mask,
)


HERE = Path(__file__).resolve().parent
DIRECTIONS = ("+x", "-x", "+y", "-y")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Epsilon-only Austria 1GHz synthetic diagnostic with fixed oracle "
            "internal total fields. No field branch, PDE, boundary, robust loss, "
            "detach mode, continuation, or neural epsilon branch."
        )
    )
    parser.add_argument("--source-data-dir", default=str(HERE / "data_1GHz"))
    parser.add_argument(
        "--output-dir",
        default=str(HERE / "results_5_3_3_austria_1GHz_synthetic_epsilon_only_oracle_field"),
    )
    parser.add_argument("--init-mode", choices=["background", "true"], required=True)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--lambda-tv", type=float, default=1.0e-4)
    parser.add_argument("--lambda-l1", type=float, default=1.0e-5)
    parser.add_argument("--incident-amplitude", type=float, default=0.1)
    parser.add_argument(
        "--background-init-eps",
        type=float,
        default=1.05,
        help="Trainable epsilon used for --init-mode background; loss(background) is still evaluated at exact eps=1.",
    )
    parser.add_argument("--max-points-per-direction", type=int, default=836)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or 'cuda:0'.")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--plot-grid-size", type=int, default=220)
    parser.add_argument("--no-subdir", action="store_true", help="Write directly into --output-dir.")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def make_grid(target, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    half = float(target.roi_half_width)
    dx = 2.0 * half / grid_size
    coords = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, grid_size)
    xx, yy = np.meshgrid(coords, coords)
    xy = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float64)
    eps = np.full((grid_size, grid_size), target.eps_background, dtype=np.float64)
    eps[target_mask(target, xx, yy)] = target.eps_object
    return xx, yy, xy, float(dx * dx), eps


def green_matrix(dst_xy: np.ndarray, src_xy: np.ndarray, k0: float) -> np.ndarray:
    delta = dst_xy[:, None, :] - src_xy[None, :, :]
    distance = np.linalg.norm(delta, axis=2).clip(min=1.0e-9)
    return 0.25j * hankel1(0, k0 * distance)


def solve_oracle_total_fields(
    *,
    cell_xy: np.ndarray,
    area_weight: float,
    eps_flat: np.ndarray,
    k0: float,
    amplitude: complex,
) -> dict[str, np.ndarray]:
    contrast = eps_flat - 1.0
    operator = green_matrix(cell_xy, cell_xy, k0) * ((k0**2) * area_weight * contrast[None, :])
    system = np.eye(cell_xy.shape[0], dtype=np.complex128) - operator
    out: dict[str, np.ndarray] = {}
    for label in DIRECTIONS:
        _, direction = parse_direction_from_name(Path(f"{label}.txt"))
        incident = incident_field_numpy(cell_xy, direction, k0, amplitude, phase_sign=1.0)
        out[label] = np.linalg.solve(system, incident)
    return out


def load_receiver_points(args: argparse.Namespace, config: TrainConfig) -> dict[str, np.ndarray]:
    source_dir = Path(args.source_data_dir)
    out: dict[str, np.ndarray] = {}
    for label in DIRECTIONS:
        xy, _ = load_fem_table(source_dir / f"{label}.txt", imag_sign=config.observation_imag_sign)
        if args.max_points_per_direction > 0 and xy.shape[0] > args.max_points_per_direction:
            keep = np.linspace(0, xy.shape[0] - 1, args.max_points_per_direction).round().astype(int)
            xy = xy[keep]
        out[label] = xy
    return out


def make_observation_tensors(
    *,
    receiver_xy: dict[str, np.ndarray],
    cell_xy: np.ndarray,
    area_weight: float,
    true_eps_flat: np.ndarray,
    oracle_total: dict[str, np.ndarray],
    config: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    k2_area = (config.k0**2) * area_weight
    source_true: dict[str, np.ndarray] = {}
    for label in DIRECTIONS:
        source_true[label] = k2_area * (true_eps_flat - 1.0) * oracle_total[label]
    for label in DIRECTIONS:
        green = green_matrix(receiver_xy[label], cell_xy, config.k0)
        scattered = green @ source_true[label]
        out[label] = {
            "green_re": torch.as_tensor(green.real, dtype=dtype, device=device),
            "green_im": torch.as_tensor(green.imag, dtype=dtype, device=device),
            "oracle_re": torch.as_tensor(oracle_total[label].real, dtype=dtype, device=device),
            "oracle_im": torch.as_tensor(oracle_total[label].imag, dtype=dtype, device=device),
            "target_re": torch.as_tensor(scattered.real, dtype=dtype, device=device),
            "target_im": torch.as_tensor(scattered.imag, dtype=dtype, device=device),
        }
    return out


def eps_to_logits(eps: np.ndarray, eps_min: float, eps_max: float) -> np.ndarray:
    p = (eps - eps_min) / (eps_max - eps_min)
    p = np.clip(p, 1.0e-5, 1.0 - 1.0e-5)
    return np.log(p / (1.0 - p))


def logits_to_eps(logits: torch.Tensor, eps_min: float, eps_max: float) -> torch.Tensor:
    return eps_min + (eps_max - eps_min) * torch.sigmoid(logits)


def tv_loss(eps: torch.Tensor) -> torch.Tensor:
    dx = eps[:, 1:] - eps[:, :-1]
    dy = eps[1:, :] - eps[:-1, :]
    return dx.abs().mean() + dy.abs().mean()


def compute_loss(
    eps: torch.Tensor,
    obs: dict[str, dict[str, torch.Tensor]],
    *,
    eps_background: float,
    k2_area: float,
    lambda_tv: float,
    lambda_l1: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    contrast = (eps.reshape(-1) - eps_background)
    data_terms = []
    for row in obs.values():
        source_re = k2_area * contrast * row["oracle_re"]
        source_im = k2_area * contrast * row["oracle_im"]
        pred_re = torch.matmul(row["green_re"], source_re) - torch.matmul(row["green_im"], source_im)
        pred_im = torch.matmul(row["green_re"], source_im) + torch.matmul(row["green_im"], source_re)
        diff_re = pred_re - row["target_re"]
        diff_im = pred_im - row["target_im"]
        data_terms.append((diff_re.square() + diff_im.square()).mean())
    data = torch.stack(data_terms).mean()
    tv = tv_loss(eps)
    l1 = contrast.abs().mean()
    total = data + lambda_tv * tv + lambda_l1 * l1
    return total, {"data": data, "tv": tv, "contrast_l1": l1}


def eval_metrics(eps_np: np.ndarray, true_eps: np.ndarray, target) -> dict[str, float]:
    return epsilon_metrics(eps_np, true_eps, target)


def grad_norm(param: torch.Tensor) -> float:
    if param.grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(param.grad.detach()).cpu())


def evaluate_fixed(
    eps_np: np.ndarray,
    obs: dict[str, dict[str, torch.Tensor]],
    *,
    target,
    true_eps: np.ndarray,
    k2_area: float,
    lambda_tv: float,
    lambda_l1: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    eps_t = torch.as_tensor(eps_np, dtype=dtype, device=device)
    with torch.no_grad():
        total, parts = compute_loss(
            eps_t,
            obs,
            eps_background=target.eps_background,
            k2_area=k2_area,
            lambda_tv=lambda_tv,
            lambda_l1=lambda_l1,
        )
    metrics = eval_metrics(eps_np, true_eps, target)
    return {
        "loss": float(total.cpu()),
        "receiver_data_loss": float(parts["data"].cpu()),
        "tv_loss": float(parts["tv"].cpu()),
        "contrast_l1": float(parts["contrast_l1"].cpu()),
        **metrics,
    }


def save_plot(xx: np.ndarray, yy: np.ndarray, true_eps: np.ndarray, recon: np.ndarray, target, out_path: Path) -> None:
    plot_comparison(xx, yy, true_eps, recon, target, out_path, title="1.0 GHz epsilon-only oracle field")


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    target = build_target(0.0)
    config = TrainConfig(
        frequency_hz=1.0e9,
        incident_amplitude=args.incident_amplitude,
        incident_phase_sign=1.0,
        observation_imag_sign=-1.0,
        eps_min=1.0,
        eps_max=4.0,
    )
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    torch.manual_seed(20260430)
    np.random.seed(20260430)

    output_root = Path(args.output_dir)
    output_dir = output_root if args.no_subdir else output_root / f"init_{args.init_mode}_grid{args.grid_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    xx, yy, cell_xy, area_weight, true_eps = make_grid(target, args.grid_size)
    true_eps_flat = true_eps.reshape(-1)
    background_eps = np.full_like(true_eps, target.eps_background)
    init_eps = (
        np.full_like(true_eps, float(args.background_init_eps))
        if args.init_mode == "background"
        else true_eps.copy()
    )

    print(f"frequency_hz={config.frequency_hz}")
    print(f"k0={config.k0}")
    print(f"grid_size={args.grid_size}")
    print(f"device={device}")
    print(f"dtype={args.dtype}")
    print(f"source_data_dir={Path(args.source_data_dir).resolve()}")
    print(f"output_dir={output_dir.resolve()}")
    print("operator=G(+i/4 H0), source(+k0^2), contrast(epsilon-1)")
    print("oracle_internal_field=true epsilon discrete Lippmann-Schwinger total field")
    print("loss=receiver_data_loss + lambda_tv * tv_loss + lambda_l1 * contrast_l1")
    print("disabled=field_branch,pde,boundary,robust,detach,continuation,neural_epsilon_branch")

    receiver_xy = load_receiver_points(args, config)
    oracle_total = solve_oracle_total_fields(
        cell_xy=cell_xy,
        area_weight=area_weight,
        eps_flat=true_eps_flat,
        k0=config.k0,
        amplitude=complex(args.incident_amplitude, 0.0),
    )
    obs = make_observation_tensors(
        receiver_xy=receiver_xy,
        cell_xy=cell_xy,
        area_weight=area_weight,
        true_eps_flat=true_eps_flat,
        oracle_total=oracle_total,
        config=config,
        device=device,
        dtype=dtype,
    )
    k2_area = (config.k0**2) * area_weight

    fixed_losses = {
        "true": evaluate_fixed(
            true_eps,
            obs,
            target=target,
            true_eps=true_eps,
            k2_area=k2_area,
            lambda_tv=args.lambda_tv,
            lambda_l1=args.lambda_l1,
            device=device,
            dtype=dtype,
        ),
        "background": evaluate_fixed(
            background_eps,
            obs,
            target=target,
            true_eps=true_eps,
            k2_area=k2_area,
            lambda_tv=args.lambda_tv,
            lambda_l1=args.lambda_l1,
            device=device,
            dtype=dtype,
        ),
        "init": evaluate_fixed(
            init_eps,
            obs,
            target=target,
            true_eps=true_eps,
            k2_area=k2_area,
            lambda_tv=args.lambda_tv,
            lambda_l1=args.lambda_l1,
            device=device,
            dtype=dtype,
        ),
    }
    print("pretrain_losses")
    for name, row in fixed_losses.items():
        print(
            f"{name}: loss={row['loss']:.8e} data={row['receiver_data_loss']:.8e} "
            f"tv={row['tv_loss']:.8e} l1={row['contrast_l1']:.8e} "
            f"rel={row['rel_error_continuous']:.6f} ssim={row['ssim_continuous']:.6f} "
            f"rel_thr={row['rel_error_thresholded']:.6f} ssim_thr={row['ssim_thresholded']:.6f}",
            flush=True,
        )

    logits_np = eps_to_logits(init_eps, config.eps_min, config.eps_max)
    logits = torch.nn.Parameter(torch.as_tensor(logits_np, dtype=dtype, device=device))
    optimizer = torch.optim.Adam([logits], lr=args.lr)
    history: list[dict[str, float]] = []

    for step in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        eps = logits_to_eps(logits, config.eps_min, config.eps_max)
        total, parts = compute_loss(
            eps,
            obs,
            eps_background=target.eps_background,
            k2_area=k2_area,
            lambda_tv=args.lambda_tv,
            lambda_l1=args.lambda_l1,
        )
        total.backward()
        gnorm = grad_norm(logits)
        optimizer.step()

        should_log = step == 1 or step % args.log_every == 0 or step == args.epochs
        if should_log:
            eps_np = eps.detach().cpu().numpy()
            metric_items = eval_metrics(eps_np, true_eps, target)
            row = {
                "step": float(step),
                "loss": float(total.detach().cpu()),
                "receiver_data_loss": float(parts["data"].detach().cpu()),
                "tv_loss": float(parts["tv"].detach().cpu()),
                "contrast_l1": float(parts["contrast_l1"].detach().cpu()),
                "eps_min": float(eps_np.min()),
                "eps_max": float(eps_np.max()),
                "eps_mean": float(eps_np.mean()),
                "grad_norm": gnorm,
                "elapsed_s": float(time.time() - start),
                **metric_items,
            }
            history.append(row)
            print(
                "step={step:6.0f} loss={loss:.8e} data={receiver_data_loss:.8e} "
                "tv={tv_loss:.8e} l1={contrast_l1:.8e} "
                "eps=[{eps_min:.4f},{eps_max:.4f}] mean={eps_mean:.4f} "
                "grad={grad_norm:.4e} rel={rel_error_continuous:.6f} "
                "ssim={ssim_continuous:.6f} rel_thr={rel_error_thresholded:.6f} "
                "ssim_thr={ssim_thresholded:.6f}".format(**row),
                flush=True,
            )

    final_eps = logits_to_eps(logits, config.eps_min, config.eps_max).detach().cpu().numpy()
    final_metrics = evaluate_fixed(
        final_eps,
        obs,
        target=target,
        true_eps=true_eps,
        k2_area=k2_area,
        lambda_tv=args.lambda_tv,
        lambda_l1=args.lambda_l1,
        device=device,
        dtype=dtype,
    )
    final_metrics.update(
        {
            "elapsed_s": float(time.time() - start),
            "init_mode": args.init_mode,
            "grid_size": args.grid_size,
            "lr": args.lr,
            "lambda_tv": args.lambda_tv,
            "lambda_l1": args.lambda_l1,
            "frequency_hz": config.frequency_hz,
            "k0": config.k0,
            "area_weight": area_weight,
            "pretrain_losses": fixed_losses,
            "target": asdict(target),
            "output_dir": str(output_dir),
        }
    )

    np.save(output_dir / "final_epsilon.npy", final_eps)
    np.save(output_dir / "true_epsilon.npy", true_eps)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    if history:
        with (output_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    save_plot(xx, yy, true_eps, final_eps, target, output_dir / "true_vs_recon.png")
    print(json.dumps({"metrics": final_metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
