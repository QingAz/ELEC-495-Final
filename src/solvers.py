from __future__ import annotations

import torch

from .systems import ForcedDampedOscillator


def euler_step(system: ForcedDampedOscillator, t: torch.Tensor, z: torch.Tensor, dt: float) -> torch.Tensor:
    return z + dt * system.derivative(t, z)


def rk4_step(system: ForcedDampedOscillator, t: torch.Tensor, z: torch.Tensor, dt: float) -> torch.Tensor:
    dt_t = torch.tensor(dt, device=z.device, dtype=z.dtype)
    k1 = system.derivative(t, z)
    k2 = system.derivative(t + 0.5 * dt_t, z + 0.5 * dt * k1)
    k3 = system.derivative(t + 0.5 * dt_t, z + 0.5 * dt * k2)
    k4 = system.derivative(t + dt_t, z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def backward_euler_step(system: ForcedDampedOscillator, t: torch.Tensor, z: torch.Tensor, dt: float) -> torch.Tensor:
    t_next = t + torch.tensor(dt, device=z.device, dtype=z.dtype)
    b_next = torch.stack(
        (torch.zeros_like(t_next), system.force_amp * torch.sin(system.force_freq * t_next)),
        dim=-1,
    )
    rhs = z + dt * b_next

    m = system.be_matrix(dt=dt, device=z.device, dtype=z.dtype)
    if rhs.ndim == 1:
        return torch.linalg.solve(m, rhs)
    m_batch = m.unsqueeze(0).expand(rhs.shape[0], -1, -1)
    return torch.linalg.solve(m_batch, rhs)


def one_step(method: str, system: ForcedDampedOscillator, t: torch.Tensor, z: torch.Tensor, dt: float) -> torch.Tensor:
    if method == "euler":
        return euler_step(system, t, z, dt)
    if method == "rk4":
        return rk4_step(system, t, z, dt)
    if method == "backward_euler":
        return backward_euler_step(system, t, z, dt)
    raise ValueError(f"Unknown method: {method}")


def rollout_solver(
    method: str,
    system: ForcedDampedOscillator,
    z0: torch.Tensor,
    t0: float,
    t_end: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_steps = int(round((t_end - t0) / dt))
    t = torch.linspace(t0, t0 + n_steps * dt, n_steps + 1, device=z0.device, dtype=z0.dtype)
    z = torch.zeros((n_steps + 1, 2), device=z0.device, dtype=z0.dtype)
    z[0] = z0
    for i in range(n_steps):
        z[i + 1] = one_step(method, system, t[i], z[i], dt)
    return t, z
