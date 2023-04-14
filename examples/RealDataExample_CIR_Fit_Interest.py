"""
Description: This example demonstrates how to fit an SDE based on historical interest rate data.
The data consists of daily observations of the 10 Year Constant maturity interst rates.
The example fits three Maximum Likelihood Estimators (MLE):
    1) Shoji-Ozaki,
    2) Kessler's approximation
    3) Elerian Density approximation

Data Source:
-----------
Board of Governors of the Federal Reserve System (US), 10-Year Treasury Constant Maturity Rate [DGS10],
retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/DGS10, April 11, 2021.

"""
import matplotlib.pyplot as plt
from pymle.models.CIR import CIR
from pymle.core.TransitionDensity import *
from pymle.fit.AnalyticalMLE import AnalyticalMLE
import seaborn as sns
import matplotlib.dates as mdates
from pymle.data.loader import load_FX_USD_EUR, load_10yr_CMrate
import datetime

sns.set_style('whitegrid')

# ===========================
# Create the Hypothesized model (CIR)
# ===========================
model = CIR()
guess = np.asarray([.24, 1.0, 0.1])
param_bounds = [(0.01, 5), (0.01, 2), (0.01, 0.9)]
# ===========================
# Read in the data (FX time series)
# ===========================
example = 2
if example == 1:
    df = load_FX_USD_EUR()
else:
    df = load_10yr_CMrate()

# =====================
# Plot the data
# =====================
skip = 20
dt = skip / 252.
sample = df['Rate'].values[:-1:skip]

do_plot = True

if do_plot:
    fig, ax = plt.subplots()

    df['Date'] = [datetime.datetime.strptime(d, "%m/%d/%Y").date() for d in df['Date']]

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax.plot(df['Date'].values, df['Rate'].values)
    plt.xlabel('Date')
    plt.ylabel('Rate')
    fig.autofmt_xdate()

    plt.show()

# ===========================
# Fit maximum Likelihood estimators
# ===========================

# Fit using Exact MLE
exact_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                          density=ExactDensity(model)).estimate_params(guess)

print(f'\nExact MLE: {exact_est}')

# Fit using Kessler MLE
print("\nStarting KESSLER------------------\n")
kessler_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=KesslerDensity(model)).estimate_params(guess)
print(f'\nKessler MLE: {kessler_est} \n')

# Fit using Elerian MLE
print("\nStarting ELERIAN------------------\n")
kessler_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=ElerianDensity(model)).estimate_params(guess)

# Fit using Shoji-Ozaki MLE
print("\nStarting SHOJI-OZAKI------------------\n")
shojioz_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                            density=ShojiOzakiDensity(model)).estimate_params(guess)
print(f'\nShoji-Ozaki MLE: {shojioz_est}')

# Fit using Ozaki
print("\nStarting OZAKI------------------\n")
exact_est = AnalyticalMLE(sample=sample, param_bounds=param_bounds, dt=dt,
                          density=OzakiDensity(model)).estimate_params(guess)

print(f'\nOzaki MLE: {exact_est}')