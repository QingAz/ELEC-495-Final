from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import simulate_trajectory
from .models import NeuralODEWrapper, ResNetStepper, TimeToStateModel, VectorFieldMLP
from .plots import plot_rollout_drift, plot_rollout_traces, plot_solver_errors
from .solvers import rollout_solver
from .systems import build_system
from .utils import ensure_dir, get_device, load_yaml, project_root, save_json, set_seed


def _interp_reference(t_ref: np.ndarray, z_ref: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    x_interp = np.interp(t_target, t_ref, z_ref[:, 0])
    v_interp = np.interp(t_target, t_ref, z_ref[:, 1])
    return np.stack([x_interp, v_interp], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--dt-config", type=str, default="configs/dt_sweep.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dt_cfg = load_yaml(args.dt_config)

    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])

    root = project_root()
    fig_dir = root / "results" / "figures"
    log_dir = root / "results" / "logs"
    ckpt_dir = root / "results" / "checkpoints"
    ensure_dir(fig_dir)
    ensure_dir(log_dir)

    system = build_system(cfg["system"])
    t_end = float(cfg["eval"]["rollout_seconds"])
    z0 = torch.tensor(cfg["eval"]["z0_eval"], dtype=torch.float32, device=device)

    # 1) Numerical solver error + stability
    ref_dt = float(cfg["eval"]["solver_ref_dt"])
    t_ref, z_ref = rollout_solver("rk4", system, z0, 0.0, t_end, ref_dt)
    t_ref_np = t_ref.detach().cpu().numpy()
    z_ref_np = z_ref.detach().cpu().numpy()

    dt_values = dt_cfg.get("sweep", {}).get("dt_values", cfg["eval"]["solver_dts"])
    methods = ["euler", "rk4", "backward_euler"]
    stable_threshold = float(cfg["eval"]["stable_threshold"])

    solver_errors = {m: [] for m in methods}
    solver_stability = {m: [] for m in methods}

    for m in methods:
        for dt in dt_values:
            t_m, z_m = rollout_solver(m, system, z0, 0.0, t_end, float(dt))
            t_np = t_m.detach().cpu().numpy()
            z_np = z_m.detach().cpu().numpy()
            z_ref_aligned = _interp_reference(t_ref_np, z_ref_np, t_np)
            rmse = float(np.sqrt(np.mean((z_np - z_ref_aligned) ** 2)))
            max_norm = float(np.max(np.linalg.norm(z_np, axis=1)))
            stable = bool(np.isfinite(z_np).all() and max_norm < stable_threshold)
            solver_errors[m].append((float(dt), rmse))
            solver_stability[m].append((float(dt), stable, max_norm))

    # 2) Learned model rollout drift
    # Truth at model dt
    model_dt = float(cfg["data"]["dt"])
    t_eval, z_truth = simulate_trajectory(system, z0, t_end=t_end, dt=model_dt, method="rk4")

    hidden = int(cfg["train"]["hidden"])
    depth = int(cfg["train"]["depth"])

    ff = TimeToStateModel(hidden=hidden, depth=depth).to(device)
    ff.load_state_dict(torch.load(ckpt_dir / "ff_model.pt", map_location=device))
    ff.eval()

    stepper = ResNetStepper(hidden=hidden, depth=depth).to(device)
    stepper.load_state_dict(torch.load(ckpt_dir / "resnet_stepper.pt", map_location=device))
    stepper.eval()

    vf = VectorFieldMLP(
        hidden=hidden,
        depth=depth,
        use_drive_features=bool(cfg["train"].get("ode_drive_features", True)),
        force_freq=float(cfg["system"]["force_freq"]),
    ).to(device)
    vf.load_state_dict(torch.load(ckpt_dir / "neural_ode_vf.pt", map_location=device))
    vf.eval()
    n_ode = NeuralODEWrapper(vf).to(device)

    with torch.no_grad():
        z_ff = ff(t_eval.unsqueeze(-1))
        z_step = stepper.rollout(z0=z_truth[0], time=t_eval)
        z_node = n_ode.rollout(z0=z_truth[0], time=t_eval, method="rk4")

    truth_np = z_truth.detach().cpu().numpy()
    pred_np = {
        "feedforward_t2z": z_ff.detach().cpu().numpy(),
        "resnet_stepper": z_step.detach().cpu().numpy(),
        "neural_ode": z_node.detach().cpu().numpy(),
    }

    t_np = t_eval.detach().cpu().numpy()
    drifts = {
        k: np.linalg.norm(v - truth_np, axis=1) for k, v in pred_np.items()
    }

    metrics = {
        "solver_errors": {k: [[dt, err] for dt, err in vals] for k, vals in solver_errors.items()},
        "solver_stability": {
            k: [[dt, stable, max_norm] for dt, stable, max_norm in vals] for k, vals in solver_stability.items()
        },
        "rollout_final_drift": {k: float(v[-1]) for k, v in drifts.items()},
        "rollout_mean_drift": {k: float(np.mean(v)) for k, v in drifts.items()},
    }
    save_json(log_dir / "eval_metrics.json", metrics)

    plot_solver_errors(solver_errors, fig_dir / "solver_error_vs_dt.png")
    plot_rollout_drift(t_np, drifts, fig_dir / "rollout_drift.png")
    plot_rollout_traces(t_np, truth_np, pred_np, fig_dir / "rollout_traces.png")


if __name__ == "__main__":
    main()
