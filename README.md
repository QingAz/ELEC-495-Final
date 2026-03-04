# A1-oscillator-siml

A1 theme: damped/forced oscillator, comparing numerical stability vs learned dynamics.

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

## Quick start
```bash
pip install -r requirements.txt
python -m src.train --config configs/base.yaml
python -m src.eval --config configs/base.yaml --dt-config configs/dt_sweep.yaml
```

Or run:
```bash
bash scripts/run_all.sh
```

Outputs:
- `results/checkpoints/`: trained model checkpoints
- `results/logs/`: JSON metrics
- `results/figures/`: comparison figures
