"""Post-process the official Re=1000 LDC PINN prediction.

This script keeps the official model unchanged and adds a small, reproducible
error-diagnosis step: reference/prediction/error fields plus separate metrics
for the near-wall and core regions of the square cavity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import paddle
import scipy.io as sio

import ppsci
from ppsci.utils import misc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--near-wall-width",
        type=float,
        default=0.1,
        help="Near-wall width as a fraction of the cavity side length.",
    )
    return parser.parse_args()


def region_metrics(
    prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    pred = prediction[mask].astype(np.float64)
    ref = reference[mask].astype(np.float64)
    error = pred - ref
    return {
        "point_count": int(mask.sum()),
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(ref)),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "p95_absolute_error": float(np.quantile(np.abs(error), 0.95)),
        "p99_absolute_error": float(np.quantile(np.abs(error), 0.99)),
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def make_model() -> ppsci.arch.ModifiedMLP:
    return ppsci.arch.ModifiedMLP(
        input_keys=("x", "y"),
        output_keys=("u", "v", "p"),
        num_layers=5,
        hidden_size=256,
        activation="tanh",
        fourier={"scale": 10.0, "dim": 128},
        random_weight={"mean": 1.0, "std": 0.1},
    )


def save_comparison_figure(
    reference: np.ndarray,
    prediction: np.ndarray,
    absolute_error: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path,
) -> None:
    field_min = float(min(reference.min(), prediction.min()))
    field_max = float(max(reference.max(), prediction.max()))
    error_color_cap = float(np.quantile(absolute_error, 0.995))
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    panels = (
        (reference, "Reference velocity magnitude", "jet", field_min, field_max),
        (prediction, "PINN prediction", "jet", field_min, field_max),
        (
            absolute_error,
            f"Absolute error (color capped at p99.5={error_color_cap:.3f})",
            "magma",
            0.0,
            error_color_cap,
        ),
    )
    for axis, (field, title, cmap, vmin, vmax) in zip(axes, panels):
        image = axis.imshow(
            field.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("Lid-driven cavity flow at Re=1000")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paddle.set_device(args.device)

    data = sio.loadmat(args.data)
    dtype = paddle.get_default_dtype()
    u_ref = data["u"].astype(dtype)
    v_ref = data["v"].astype(dtype)
    speed_ref = np.sqrt(u_ref**2 + v_ref**2)
    x = data["x"].ravel().astype(dtype)
    y = data["y"].ravel().astype(dtype)

    xy = misc.cartesian_product(x, y).astype(dtype)
    inputs = {"x": xy[:, 0:1], "y": xy[:, 1:2]}
    solver = ppsci.solver.Solver(
        make_model(),
        device=args.device,
        pretrained_model_path=str(args.weights),
        eval_with_no_grad=True,
    )
    prediction = solver.predict(
        inputs, batch_size=args.batch_size, return_numpy=True
    )
    u_pred = prediction["u"].reshape(len(x), len(y))
    v_pred = prediction["v"].reshape(len(x), len(y))
    speed_pred = np.sqrt(u_pred**2 + v_pred**2)
    absolute_error = np.abs(speed_pred - speed_ref)

    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    side_length = min(float(x.max() - x.min()), float(y.max() - y.min()))
    wall_distance = np.minimum.reduce(
        (
            grid_x - x.min(),
            x.max() - grid_x,
            grid_y - y.min(),
            y.max() - grid_y,
        )
    )
    near_wall_limit = args.near_wall_width * side_length
    near_wall_mask = wall_distance <= near_wall_limit
    core_mask = ~near_wall_mask
    full_mask = np.ones_like(near_wall_mask, dtype=bool)

    metrics = {
        "case": "LDC2D steady Re=1000 official pretrained model",
        "device": paddle.device.get_device(),
        "grid_shape": [int(len(x)), int(len(y))],
        "near_wall_definition": {
            "fraction_of_side_length": float(args.near_wall_width),
            "distance_threshold": float(near_wall_limit),
        },
        "global": region_metrics(speed_pred, speed_ref, full_mask),
        "near_wall": region_metrics(speed_pred, speed_ref, near_wall_mask),
        "core": region_metrics(speed_pred, speed_ref, core_mask),
    }

    metrics_path = args.output_dir / "region_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "velocity_fields.npz",
        x=x,
        y=y,
        u_reference=u_ref,
        v_reference=v_ref,
        speed_reference=speed_ref,
        u_prediction=u_pred,
        v_prediction=v_pred,
        speed_prediction=speed_pred,
        absolute_error=absolute_error,
        near_wall_mask=near_wall_mask,
    )
    save_comparison_figure(
        speed_ref,
        speed_pred,
        absolute_error,
        x,
        y,
        args.output_dir / "reference_prediction_error.png",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved figure to {args.output_dir / 'reference_prediction_error.png'}")


if __name__ == "__main__":
    main()
