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





