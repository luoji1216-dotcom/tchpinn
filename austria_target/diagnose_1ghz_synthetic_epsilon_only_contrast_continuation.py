from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "square_target"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from austria_optimized_common import build_target  # noqa: E402
from diagnose_1ghz_synthetic_epsilon_only_oracle_field import (  # noqa: E402
    compute_loss,
    eps_to_logits,
    eval_metrics,
    load_receiver_points,
    logits_to_eps,
    make_grid,
    make_observation_tensors,
    resolve_device,
    resolve_dtype,
    save_plot,
    solve_oracle_total_fields,
)
from pinn_pixel_inverse_core import TrainConfig  # noqa: E402


HERE = Path(__file__).resolve().parent
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Contrast continuation for 1GHz synthetic epsilon-only oracle-field inversion."
    )
    parser.add_argument("--source-data-dir", default=str(HERE / "data_1GHz"))
    parser.add_argument(
        "--output-dir",
        default=str(HERE / "results_5_3_3_austria_1GHz_synthetic_epsilon_only_contrast_continuation"),
    )
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--epochs-per-alpha", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--lambda-tv", type=float, default=1.0e-4)
    parser.add_argument("--lambda-l1", type=float, default=1.0e-5)
    parser.add_argument(
        "--alphas",
        default=",".join(f"{value:g}" for value in ALPHAS),
        help="Comma-separated contrast continuation alpha schedule.",
    )
    parser.add_argument("--background-init-eps", type=float, default=1.05)
    parser.add_argument("--incident-amplitude", type=float, default=0.1)
    parser.add_argument("--max-points-per-direction", type=int, default=836)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--log-every", type=int, default=300)
    return parser


def parse_alphas(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--alphas must contain at least one value.")
    if any(value <= 0.0 for value in values):
        raise ValueError("--alphas values must be positive.")
    if any(values[i] <= values[i - 1] for i in range(1, len(values))):
        raise ValueError("--alphas must be strictly increasing.")
    return values


def grad_norm(param: torch.Tensor) -> float:
    if param.grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(param.grad.detach()).cpu())


def train_stage(
    *,
    alpha: float,
    init_eps: np.ndarray,
    args: argparse.Namespace,
    config: TrainConfig,
    full_target,
    receiver_xy,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    stage_target = replace(full_target, eps_object=1.0 + alpha * (full_target.eps_object - 1.0))
    xx, yy, cell_xy, area_weight, stage_true_eps = make_grid(stage_target, args.grid_size)
    stage_true_flat = stage_true_eps.reshape(-1)
    oracle_total = solve_oracle_total_fields(
        cell_xy=cell_xy,
        area_weight=area_weight,
        eps_flat=stage_true_flat,
        k0=config.k0,
        amplitude=complex(args.incident_amplitude, 0.0),
    )
    obs = make_observation_tensors(
        receiver_xy=receiver_xy,
        cell_xy=cell_xy,
        area_weight=area_weight,
        true_eps_flat=stage_true_flat,
        oracle_total=oracle_total,
        config=config,
        device=device,
        dtype=dtype,
    )
    k2_area = (config.k0**2) * area_weight
    stage_dir = output_dir / f"alpha_{alpha:.2f}".replace(".", "p")
    stage_dir.mkdir(parents=True, exist_ok=True)

    logits = torch.nn.Parameter(
        torch.as_tensor(eps_to_logits(init_eps, config.eps_min, config.eps_max), dtype=dtype, device=device)
    )
    optimizer = torch.optim.Adam([logits], lr=args.lr)
    history: list[dict[str, float]] = []
    start = time.time()

    print(f"\n=== alpha={alpha:.2f} eps_object={stage_target.eps_object:.6g} ===", flush=True)
    for step in range(1, args.epochs_per_alpha + 1):
        optimizer.zero_grad(set_to_none=True)
        eps = logits_to_eps(logits, config.eps_min, config.eps_max)
        total, parts = compute_loss(
            eps,
            obs,
            eps_background=stage_target.eps_background,
            k2_area=k2_area,
            lambda_tv=args.lambda_tv,
            lambda_l1=args.lambda_l1,
        )
        total.backward()
        gnorm = grad_norm(logits)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.epochs_per_alpha:
            eps_np = eps.detach().cpu().numpy()
            metrics = eval_metrics(eps_np, stage_true_eps, stage_target)
            row = {
                "alpha": float(alpha),
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
                **metrics,
            }
            history.append(row)
            print(
                "alpha={alpha:.2f} step={step:6.0f} loss={loss:.8e} "
                "data={receiver_data_loss:.8e} eps=[{eps_min:.4f},{eps_max:.4f}] "
                "mean={eps_mean:.4f} rel={rel_error_continuous:.6f} "
                "ssim={ssim_continuous:.6f} rel_thr={rel_error_thresholded:.6f} "
                "ssim_thr={ssim_thresholded:.6f}".format(**row),
                flush=True,
            )

    final_eps = logits_to_eps(logits, config.eps_min, config.eps_max).detach().cpu().numpy()
    final_metrics = eval_metrics(final_eps, stage_true_eps, stage_target)
    with torch.no_grad():
        final_loss, final_parts = compute_loss(
            torch.as_tensor(final_eps, dtype=dtype, device=device),
            obs,
            eps_background=stage_target.eps_background,
            k2_area=k2_area,
            lambda_tv=args.lambda_tv,
            lambda_l1=args.lambda_l1,
        )
    summary: dict[str, object] = {
        "alpha": float(alpha),
        "eps_object": float(stage_target.eps_object),
        "loss": float(final_loss.cpu()),
        "receiver_data_loss": float(final_parts["data"].cpu()),
        "tv_loss": float(final_parts["tv"].cpu()),
        "contrast_l1": float(final_parts["contrast_l1"].cpu()),
        "eps_min": float(final_eps.min()),
        "eps_max": float(final_eps.max()),
        "eps_mean": float(final_eps.mean()),
        **final_metrics,
        "image_path": str(stage_dir / "true_vs_recon.png"),
        "stage_dir": str(stage_dir),
    }
    np.save(stage_dir / "final_epsilon.npy", final_eps)
    np.save(stage_dir / "true_epsilon_alpha.npy", stage_true_eps)
    with (stage_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if history:
        with (stage_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    save_plot(xx, yy, stage_true_eps, final_eps, stage_target, stage_dir / "true_vs_recon.png")
    return final_eps, summary


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(20260430)
    np.random.seed(20260430)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    full_target = build_target(0.0)
    config = TrainConfig(
        frequency_hz=1.0e9,
        incident_amplitude=args.incident_amplitude,
        incident_phase_sign=1.0,
        observation_imag_sign=-1.0,
        eps_min=1.0,
        eps_max=4.0,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receiver_xy = load_receiver_points(args, config)
    init_eps = np.full((args.grid_size, args.grid_size), float(args.background_init_eps), dtype=np.float64)
    alphas = parse_alphas(args.alphas)

    print(f"frequency_hz={config.frequency_hz}")
    print(f"k0={config.k0}")
    print(f"grid_size={args.grid_size}")
    print(f"device={device}")
    print(f"output_dir={output_dir.resolve()}")
    print("operator=G(+i/4 H0), source(+k0^2), contrast(epsilon-1)")
    print("contrast_continuation_alphas=" + ",".join(f"{a:.2f}" for a in alphas))

    summaries = []
    current = init_eps
    for alpha in alphas:
        current, summary = train_stage(
            alpha=alpha,
            init_eps=current,
            args=args,
            config=config,
            full_target=full_target,
            receiver_xy=receiver_xy,
            device=device,
            dtype=dtype,
            output_dir=output_dir,
        )
        summaries.append(summary)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "target": asdict(full_target), "stages": summaries}, f, indent=2, ensure_ascii=False)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print("\ncontrast_continuation_summary")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
