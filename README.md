
# PyMLE

Python library for Maximum Likelihood estimation (MLE) and simulation of Stochastic Differntial Equations (SDE), i.e. continuous diffusion processes.

This library supports many models out of the box (e.g. Brownian Motion, Geometric Brownian Motion, CKLS, CIR, OU, etc.), and is designed to make it easy to add new models with minimial code, and to inheret the fitting and simulation of these models for free.

MLE procedures supported: Kessler, Euler, Shoji-Ozaki, Elerian, and Exact MLE (when applicable)
Simulation schemes supported: Euler, Milstein, Milstein-2


## User installation


Coming soon, for now use git clone.

## Dependencies


pymle requires:

- Python (>= 3.7)
- NumPy (tested with 1.20.2)


## Source code


You can check the latest sources with the command

    git clone https://github.com/jkirkby3/pymle.git
    
    

## Example: Fit MLE to Simulated Cox-Ingersol-Ross (CIR) sample

```python
from pymle.models import CIR
from pymle.sim.Simulator1D import Simulator1D
from pymle.TransitionDensity import ExactDensity, KesslerDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
import numpy as np

# ===========================
# Set the true model (CIR) params, to simulate the process
# ===========================
S0 = 0.4  # initial value of process

kappa = 3  # rate of mean reversion
mu = 0.3  # long term level of process
sigma = 0.2  # volatility

# ===========================
# Create the true model to fit to
# ===========================
model = CIR()
model.params = np.array([kappa, mu, sigma])

# ===========================
# Simulate a sample path (we will fit to this path)
# ===========================
T = 5  # num years of the sample
freq = 250  # observations per year
dt = 1. / freq
seed = 123  # random seed: set to None to get new results each time

simulator = Simulator1D(S0=S0, M=T * freq, dt=dt, model=model).set_seed(seed=seed)
sample = simulator.sim_path()

# ===========================
# Fit maximum Likelihood estimators
# ===========================
# Set the parameter bounds for fitting  (kappa, mu, sigma)
param_bounds = [(0, 10), (0, 4), (0.01, 1)]

# Choose some initial guess for params fit
guess = np.array([1, 0.1, 0.4])

# Fit using Kessler MLE
kessler_est = AnalyticalMLE(sample, param_bounds, dt, density=KesslerDensity(model)).estimate_params(guess)

print(f'\nKessler MLE: {kessler_est} \n')

# Fit using Exact MLE
exact_est = AnalyticalMLE(sample, param_bounds, dt, density=ExactDensity(model)).estimate_params(guess)

print(f'\nExact MLE: {exact_est}')
```

```
Kessler MLE: 
params     | [2.9628751  0.3157062  0.19978367] 
likelihood | 4398.276719373996 
AIC        | -8790.553438747991
BIC        | -8775.160742257101 

Exact MLE: 
params     | [3.0169684  0.31590483 0.2011907 ] 
likelihood | 4397.641069883833 
AIC        | -8789.282139767665
BIC        | -8773.889443276776

```

