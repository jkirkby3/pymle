
# PyMLE

Python library for Maximum Likelihood estimation (MLE) and simulation of Stochastic Differntial Equations (SDE), i.e. continuous diffusion processes.

This library supports many models out of the box (e.g. Brownian Motion, Geometric Brownian Motion, CKLS, CIR, OU, etc.), and is designed to make it easy to add new models with minimial code, and to inheret the fitting and simulation of these models for free.

**MLE procedures**: 
Ait-Sahalia Hermite Polynomial Expansion, Kessler, Euler, Shoji-Ozaki, Elerian, and Exact MLE (when applicable)

**Simulation schemes**: Euler, Milstein, Milstein-2, Exact (when available)

**Diffusion Models**: Brownian motion, Geometric Brownian motion (GBM),
IGBM, Peral-Verhulst, Linear SDE, Logistic, 3/2, CEV, CIR, CKLS, Feller's square root, 
Hyperbolic, Hyperbolic 2, Jacobi, Modified CIR, OU, Radial OU, Pearson, 
Nonlinear Mean Reversion, flexible Nonlinear SDE, Custom models easily supported


## Origin and Future Goals

This library was designed from a research collaboration with J.L. Kirkby, Dang H. Nguyen, Duy Nguyen, and Nhu N. Nguyen.  The goal is to provide a wide set of functionality for python users to simulate and estimate SDEs, as well as estimation tools for related statistical problems. The starting point is standard / state-of-the-art MLE estimation procedures, to be followed up with more cutting edge approaches.

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
model = CIR()  # Cox-Ingersol-Ross 

kappa = 3  # rate of mean reversion
mu = 0.3  # long term level of process
sigma = 0.2  # volatility

model.params = np.array([kappa, mu, sigma])

# ===========================
# Simulate a sample path (we will fit to this path)
# ===========================
S0 = 0.4  # initial value of process
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
params      | [2.9628751  0.3157062  0.19978367] 
sample size | 1250
likelihood  | 4398.276719373996 
AIC         | -8790.553438747991
BIC         | -8775.160742257101 

Exact MLE: 
params      | [3.0169684  0.31590483 0.2011907 ] 
sample size | 1250
likelihood  | 4397.641069883833 
AIC         | -8789.282139767665
BIC         | -8773.889443276776

```


## Example: Fit MLE to Historical Interest Rates (10 Year Constant maturity)

Data Source:

Board of Governors of the Federal Reserve System (US), 10-Year Treasury Constant Maturity Rate [DGS10],
retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DGS10, April 11, 2021.


```python

import pandas as pd
import numpy as np
from pymle.models.CKLS import CKLS
from pymle.TransitionDensity import KesslerDensity, ShojiOzakiDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE

# ===========================
# Create the Hypothesized model (CKLS)
# ===========================
# (Chan, Karolyi, Longstaff and Sanders Model)

model = CKLS() 

# Set bounds for param search, and some psuedo-reasonable initial guess
param_bounds = [(0.0, 10), (0.0, 10), (0.01, 3), (0.1, 2)]  
guess = np.array([0.01, 0.1, 0.2, 0.6])  

# ===========================
# Read in the data (interest rate time series)
# ===========================

df = pd.read_csv("../data/10yrCMrate.csv")
sample = df['Rate'].values
dt = 1. / 252  # Daily observations

# ===========================
# Fit maximum Likelihood estimators
# ===========================

# Fit using Kessler MLE
kessler_est = AnalyticalMLE(sample, param_bounds, dt, density=KesslerDensity(model)).estimate_params(guess)

print(f'\nKessler MLE: {kessler_est} \n')

# Fit using Shoji-Ozaki MLE
shojioz_est = AnalyticalMLE(sample, param_bounds, dt, density=ShojiOzakiDensity(model)).estimate_params(guess)

print(f'\nShoji-Ozaki MLE: {shojioz_est}')

```

```
Kessler MLE: 
params      | [0.07140132 0.00326561 0.56032624 0.33638139] 
sample size | 14801 
likelihood  | 20274.894525560834 
AIC         | -40541.78905112167
BIC         | -40511.37925102152 

Shoji-Ozaki MLE: 
params      | [1.89153215e-02 2.13208446e-05 5.58760343e-01 3.37675286e-01] 
sample size | 14801 
likelihood  | 20275.049010989227 
AIC         | -40542.098021978454
BIC         | -40511.68822187831

```
