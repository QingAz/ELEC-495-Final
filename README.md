ELEC-495-Final 

Damped oscillator, comparing numerical stability vs learned dynamics.

## What this project compares
- Numerical solvers: Euler vs RK4 vs Backward Euler
- Learned dynamics:
  - Direct fit: `t -> (x, v)`
  - Learned vector field with ResNet-style stepper rollout
  - Neural ODE rollout using a learned vector field

## ODE
\[ x'' + 2\gamma x' + \omega_0^2 x = F \sin(\Omega t) \]

State form with `z=[x, v]`:
- `x' = v`
- `v' = -2*gamma*v - omega0^2*x + F*sin(Omega*t)`

## Reproducibility

- All experiments use a fixed random seed (see `configs/base.yaml`).
- Running `bash scripts/run_all.sh` will generate figures in:
  - `results/figures/solver_error_vs_dt.png`
  - `results/figures/rollout_drift.png`
  - `results/figures/rollout_traces.png`
- Logs are saved to `results/logs/run_all.log`.



