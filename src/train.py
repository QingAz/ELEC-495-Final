from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import make_dataset
from .models import NeuralODEWrapper, ResNetStepper, TimeToStateModel, VectorFieldMLP
from .systems import build_system
from .utils import ensure_dir, get_device, load_yaml, project_root, save_json, set_seed


def _train_loop(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        optimizer.zero_grad()
        loss = loss_fn(batch)
        loss.backward()
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])

    root = project_root()
    ckpt_dir = root / "results" / "checkpoints"
    log_dir = root / "results" / "logs"
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    system = build_system(cfg["system"])
    ds = make_dataset(system=system, cfg=cfg, device=device)
    time = ds["time"]
    dt = float(ds["dt"])

    train_clean = ds["train"]["clean"]
    train_noisy = ds["train"]["noisy"]

    hidden = int(cfg["train"]["hidden"])
    depth = int(cfg["train"]["depth"])
    epochs = int(cfg["train"]["epochs"])
    batch_size = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["lr"])

    metrics = {}

    # 1) Direct fit t -> (x, v), using first trajectory
    t_ff = time.unsqueeze(-1)
    z_ff = train_noisy[0]
    ff_ds = TensorDataset(t_ff, z_ff)
    ff_loader = DataLoader(ff_ds, batch_size=batch_size, shuffle=True)
    ff_model = TimeToStateModel(hidden=hidden, depth=depth).to(device)
    ff_opt = torch.optim.Adam(ff_model.parameters(), lr=lr)
    mse = nn.MSELoss()

    def ff_loss(batch):
        t_b, z_b = batch
        pred = ff_model(t_b)
        return mse(pred, z_b)

    ff_losses = []
    for _ in range(epochs):
        ff_losses.append(_train_loop(ff_model, ff_loader, ff_opt, ff_loss))
    torch.save(ff_model.state_dict(), ckpt_dir / "ff_model.pt")
    metrics["ff_final_loss"] = ff_losses[-1]

    # Build one-step pairs for stepper
    t_pairs = time[:-1]
    z_now = train_noisy[:, :-1, :]
    z_next = train_clean[:, 1:, :]

    n_traj, n_step, _ = z_now.shape
    t_flat = t_pairs.repeat(n_traj)
    z_flat = z_now.reshape(-1, 2)
    z_next_flat = z_next.reshape(-1, 2)

    pair_ds = TensorDataset(t_flat, z_flat, z_next_flat)
    pair_loader = DataLoader(pair_ds, batch_size=batch_size, shuffle=True)

    # Build short-rollout sequences for Neural ODE
    rollout_k = int(cfg["train"].get("rollout_k", 10))
    total_steps = train_clean.shape[1]
    if rollout_k >= total_steps:
        raise ValueError(f"rollout_k={rollout_k} must be < trajectory length={total_steps}")
    n_starts = total_steps - rollout_k

    t_seq = time[:n_starts]
    z_start = train_noisy[:, :n_starts, :]
    z_targets = torch.stack(
        [train_clean[:, i + 1 : i + 1 + rollout_k, :] for i in range(n_starts)],
        dim=1,
    )

    t_seq_flat = t_seq.repeat(train_noisy.shape[0])
    z_start_flat = z_start.reshape(-1, 2)
    z_targets_flat = z_targets.reshape(-1, rollout_k, 2)
    ode_ds = TensorDataset(t_seq_flat, z_start_flat, z_targets_flat)
    ode_loader = DataLoader(ode_ds, batch_size=batch_size, shuffle=True)

    # 2) ResNet-stepper (discrete rollout model)
    stepper = ResNetStepper(hidden=hidden, depth=depth).to(device)
    step_opt = torch.optim.Adam(stepper.parameters(), lr=lr)

    def step_loss(batch):
        t_b, z_b, z_next_b = batch
        pred = stepper(t_b, z_b)
        return mse(pred, z_next_b)

    step_losses = []
    for _ in range(epochs):
        step_losses.append(_train_loop(stepper, pair_loader, step_opt, step_loss))
    torch.save(stepper.state_dict(), ckpt_dir / "resnet_stepper.pt")
    metrics["stepper_final_loss"] = step_losses[-1]

    # 3) Neural ODE (vector field + integration in the loss)
    vf = VectorFieldMLP(
        hidden=hidden,
        depth=depth,
        use_drive_features=bool(cfg["train"].get("ode_drive_features", True)),
        force_freq=float(cfg["system"]["force_freq"]),
    ).to(device)
    n_ode = NeuralODEWrapper(vf).to(device)
    ode_opt = torch.optim.Adam(n_ode.parameters(), lr=lr)

    def ode_loss(batch):
        t_b, z_b, z_target_seq = batch
        pred_seq = n_ode.rollout_k(z0=z_b, t0=t_b, dt=dt, steps=rollout_k, method="rk4")
        return mse(pred_seq, z_target_seq)

    ode_losses = []
    for _ in range(epochs):
        ode_losses.append(_train_loop(n_ode, ode_loader, ode_opt, ode_loss))
    torch.save(vf.state_dict(), ckpt_dir / "neural_ode_vf.pt")
    metrics["neural_ode_final_loss"] = ode_losses[-1]

    save_json(log_dir / "train_metrics.json", metrics)


if __name__ == "__main__":
    main()
