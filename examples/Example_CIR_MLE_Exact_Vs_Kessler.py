"""
Description: This example demonstrates how to fit an SDE. It simulates a sample path from a Cox-Ingersol-Ross (CIR)
process, and then fits two Maximum Likelihood Estimators (MLE):  1) Exact MLE, 2) Kessler's approximation
"""
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
kessler_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=KesslerDensity(model)).estimate_params(guess)

print(f'\nKessler MLE: {kessler_est} \n')

# Fit using Exact MLE
exact_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                          density=ExactDensity(model)).estimate_params(guess)

print(f'\nExact MLE: {exact_est}')
