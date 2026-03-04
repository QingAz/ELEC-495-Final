from __future__ import annotations

import torch

from .solvers import rollout_solver
from .systems import ForcedDampedOscillator


def _rand_init(n: int, x0_range: list[float], v0_range: list[float], device: torch.device) -> torch.Tensor:
    xr = x0_range[1] - x0_range[0]
    vr = v0_range[1] - v0_range[0]
    x0 = x0_range[0] + xr * torch.rand(n, device=device)
    v0 = v0_range[0] + vr * torch.rand(n, device=device)
    return torch.stack((x0, v0), dim=-1)


def simulate_trajectory(
    system: ForcedDampedOscillator,
    z0: torch.Tensor,
    t_end: float,
    dt: float,
    method: str = "rk4",
) -> tuple[torch.Tensor, torch.Tensor]:
    return rollout_solver(method=method, system=system, z0=z0, t0=0.0, t_end=t_end, dt=dt)


def make_dataset(system: ForcedDampedOscillator, cfg: dict, device: torch.device) -> dict:
    dcfg = cfg["data"]
    t_end = float(dcfg["t_end"])
    dt = float(dcfg["dt"])
    noise_std = float(dcfg["noise_std"])

    n_train = int(dcfg["train_traj"])
    n_val = int(dcfg["val_traj"])
    n_test = int(dcfg["test_traj"])
    n_total = n_train + n_val + n_test

    z0s = _rand_init(
        n=n_total,
        x0_range=dcfg["x0_range"],
        v0_range=dcfg["v0_range"],
        device=device,
    )

    trajs = []
    time = None
    for i in range(n_total):
        t, z = simulate_trajectory(system=system, z0=z0s[i], t_end=t_end, dt=dt, method="rk4")
        trajs.append(z)
        if time is None:
            time = t
    clean = torch.stack(trajs, dim=0)
    noisy = clean + noise_std * torch.randn_like(clean)

    return {
        "time": time,
        "dt": dt,
        "clean": clean,
        "noisy": noisy,
        "train": {"clean": clean[:n_train], "noisy": noisy[:n_train]},
        "val": {"clean": clean[n_train : n_train + n_val], "noisy": noisy[n_train : n_train + n_val]},
        "test": {"clean": clean[n_train + n_val :], "noisy": noisy[n_train + n_val :]},
    }
