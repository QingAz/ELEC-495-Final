from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeToStateModel(nn.Module):
    def __init__(self, hidden: int, depth: int):
        super().__init__()
        self.mlp = MLP(in_dim=1, out_dim=2, hidden=hidden, depth=depth)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.mlp(t)


class VectorFieldMLP(nn.Module):
    def __init__(self, hidden: int, depth: int, use_drive_features: bool = False, force_freq: float = 1.0):
        super().__init__()
        in_dim = 5 if use_drive_features else 3
        self.mlp = MLP(in_dim=in_dim, out_dim=2, hidden=hidden, depth=depth)
        self.use_drive_features = use_drive_features
        self.force_freq = float(force_freq)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if self.use_drive_features:
            wt = self.force_freq * t
            x = torch.cat((t, z, torch.sin(wt), torch.cos(wt)), dim=-1)
        else:
            x = torch.cat((t, z), dim=-1)
        return self.mlp(x)


class ResNetStepper(nn.Module):
    def __init__(self, hidden: int, depth: int):
        super().__init__()
        self.residual = MLP(in_dim=3, out_dim=2, hidden=hidden, depth=depth)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat((t, z), dim=-1)
        return z + self.residual(inp)

    def rollout(self, z0: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((time.shape[0], 2), device=z0.device, dtype=z0.dtype)
        out[0] = z0
        for i in range(time.shape[0] - 1):
            t_i = time[i].view(1)
            out[i + 1] = self.forward(t_i, out[i].view(1, 2)).squeeze(0)
        return out


class NeuralODEWrapper(nn.Module):
    def __init__(self, vf_model: VectorFieldMLP):
        super().__init__()
        self.vf = vf_model

    def _euler(self, t: torch.Tensor, z: torch.Tensor, dt: float) -> torch.Tensor:
        return z + dt * self.vf(t, z)

    def _rk4(self, t: torch.Tensor, z: torch.Tensor, dt: float) -> torch.Tensor:
        dt_t = torch.tensor(dt, device=z.device, dtype=z.dtype)
        k1 = self.vf(t, z)
        k2 = self.vf(t + 0.5 * dt_t, z + 0.5 * dt * k1)
        k3 = self.vf(t + 0.5 * dt_t, z + 0.5 * dt * k2)
        k4 = self.vf(t + dt_t, z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(self, z0: torch.Tensor, time: torch.Tensor, method: str = "rk4") -> torch.Tensor:
        out = torch.zeros((time.shape[0], 2), device=z0.device, dtype=z0.dtype)
        out[0] = z0
        for i in range(time.shape[0] - 1):
            dt = float((time[i + 1] - time[i]).item())
            t_i = time[i].view(1, 1)
            z_i = out[i].view(1, 2)
            if method == "euler":
                out[i + 1] = self._euler(t_i, z_i, dt).squeeze(0)
            else:
                out[i + 1] = self._rk4(t_i, z_i, dt).squeeze(0)
        return out

    def rollout_k(
        self,
        z0: torch.Tensor,
        t0: torch.Tensor,
        dt: float,
        steps: int,
        method: str = "rk4",
    ) -> torch.Tensor:
        if t0.ndim == 1:
            t = t0.unsqueeze(-1)
        else:
            t = t0
        z = z0
        preds = []
        dt_t = torch.tensor(dt, device=z0.device, dtype=z0.dtype)
        for _ in range(steps):
            if method == "euler":
                z = self._euler(t, z, dt)
            else:
                z = self._rk4(t, z, dt)
            preds.append(z)
            t = t + dt_t
        return torch.stack(preds, dim=1)
