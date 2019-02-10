from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import graph_objs as go

import sys
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as tsa
import statsmodels.tsa.api as smt
import scipy.stats as scs


def plotProcess(n_samples=1_000, rho=0):
    x = w = np.random.normal(size=n_samples)
    for t in range(n_samples):
        x[t] = rho * x[t-1] + w[t]

    with plt.style.context('bmh'):
        plt.figure(figsize=(10, 3))
        plt.plot(x)
        plt.title("Rho {}\ Dickey-Fuller p-value: {}".format(rho, round(tsa.adfuller(x)[1], 3)))
        plt.show()


white_noise = np.random.normal(size = 1_000)
with plt.style.context('bmh'):
    plt.figure(figsize=(15, 5))
    plt.plot(white_noise)
    plt.show()

for rho in [0, 0.6, 0.9, 1]:
    plotProcess(rho=rho)
