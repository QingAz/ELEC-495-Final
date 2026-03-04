from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ForcedDampedOscillator:
    gamma: float
    omega0: float
    force_amp: float
    force_freq: float

    def derivative(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = z[..., 0]
        v = z[..., 1]
        dx = v
        dv = -2.0 * self.gamma * v - (self.omega0 ** 2) * x + self.force_amp * torch.sin(self.force_freq * t)
        return torch.stack((dx, dv), dim=-1)

    def be_matrix(self, dt: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # (I - dt*A), where A is the linear part in z' = A z + b(t)
        return torch.tensor(
            [[1.0, -dt], [dt * (self.omega0 ** 2), 1.0 + 2.0 * self.gamma * dt]],
            dtype=dtype,
            device=device,
        )


def build_system(cfg: dict) -> ForcedDampedOscillator:
    return ForcedDampedOscillator(
        gamma=float(cfg["gamma"]),
        omega0=float(cfg["omega0"]),
        force_amp=float(cfg["force_amp"]),
        force_freq=float(cfg["force_freq"]),
    )
