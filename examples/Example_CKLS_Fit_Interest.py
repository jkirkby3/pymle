"""
Description: This example demonstrates how to fit an SDE based on historical interest rate data.
The data consists of daily observations of the 10 Year Constant maturity interst rates.
The example fits two Maximum Likelihood Estimators (MLE):  1) Shoji-Ozaki, 2) Kessler's approximation

Data Source:
-----------
Board of Governors of the Federal Reserve System (US), 10-Year Treasury Constant Maturity Rate [DGS10],
retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DGS10, April 11, 2021.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymle.models.CKLS import CKLS
from pymle.TransitionDensity import KesslerDensity, ShojiOzakiDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE

# ===========================
# Create the Hypothesized model (CKLS)
# ===========================
model = CKLS()
param_bounds = [(0.0, 10), (0.0, 10), (0.01, 3), (0.1, 2)]  # bounds for param search
guess = np.array([0.01, 0.1, 0.2, 0.6])  # Some guess for the params

# ===========================
# Read in the data (interest rate time series)
# ===========================

df = pd.read_csv("../data/10yrCMrate.csv")
sample = df['Rate'].values
dt = 1. / 252  # Daily observations

df.plot()
plt.show()

# ===========================
# Fit maximum Likelihood estimators
# ===========================

# Fit using Kessler MLE
kessler_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=KesslerDensity(model)).estimate_params(guess)

print(f'\nKessler MLE: {kessler_est} \n')

# Fit using Shoji-Ozaki MLE
shojioz_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=ShojiOzakiDensity(model)).estimate_params(guess)

print(f'\nShoji-Ozaki MLE: {shojioz_est}')
