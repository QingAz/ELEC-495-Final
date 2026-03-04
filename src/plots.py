from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_solver_errors(errors: dict, out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for method, pairs in errors.items():
        dts = [p[0] for p in pairs]
        rmses = [p[1] for p in pairs]
        plt.plot(dts, rmses, marker="o", label=method)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("dt")
    plt.ylabel("RMSE vs reference")
    plt.title("Solver Error vs dt")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_rollout_drift(time: np.ndarray, drifts: dict, out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for name, d in drifts.items():
        plt.plot(time, d, label=name)
    plt.xlabel("time")
    plt.ylabel("L2 drift")
    plt.title("Long-Horizon Rollout Drift")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_rollout_traces(time: np.ndarray, truth: np.ndarray, preds: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(time, truth[:, 0], color="black", linewidth=2.0, label="truth")
    ax[1].plot(time, truth[:, 1], color="black", linewidth=2.0, label="truth")

    for name, pred in preds.items():
        ax[0].plot(time, pred[:, 0], label=name)
        ax[1].plot(time, pred[:, 1], label=name)

    ax[0].set_ylabel("x")
    ax[1].set_ylabel("v")
    ax[1].set_xlabel("time")
    ax[0].grid(True, alpha=0.3)
    ax[1].grid(True, alpha=0.3)
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
