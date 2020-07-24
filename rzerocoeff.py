import sys
import os
import errorHandler
import argparse
import numpy as np
from math import *
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
from MetropolisHastings import metropolisHastings, gammaDistribution, pFunc, gaussDistribution

# from MetropolisHastings import metropolisHastings
import pandas as pd

# Hàm hiện thực công thức 15:


def calPosterior(x, Beta, Gamma):
    pi = st.gamma.pdf(x, a=Beta, scale=1/Gamma)
    return np.prod(pi)

# Hàm hiện thực công thức 21:


def calCoeff(X, samples):
    rzero = 0
    pi = []
    for sample in samples:
        pi.append(calPosterior(x, sample[0], sample[1]))
    n = 0
    for sample in samples:
        if sample[1] != 0 and sample[0] != 0 and pi[n] > 0:
            rzero += (pi[n] * sample[0]/sample[1])
        n += 1
    return rzero


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    url2 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    url3 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

    df = pd.read_csv(url, error_bad_lines=False)
    df2 = pd.read_csv(url2, error_bad_lines=False)
    df3 = pd.read_csv(url3, error_bad_lines=False)
    # Chọn các quốc gia: France Germany, Italy
    indexes1 = [116, 120, 137]
    indexes2 = [116, 120, 137]
    indexes3 = [108, 112, 131]
    coeffs = list()
    for i in range(len(indexes1)):
        df_ = df.iloc[indexes1[i]]
        df2_ = df2.iloc[indexes2[i]]
        df3_ = df3.iloc[indexes3[i]]
        X = df_[4:].values
        X2 = df2_[4:].values
        X3 = df3_[4:].values
        X1 = X - X2 - X3  # infected = confirmed - recovered - death

        x = []
        for i in range(len(X)):
            s = (X1[i] + X3[i])
            if s > 0:
                x.append(s)
        samples = metropolisHastings(10000, gammaDistribution)
        result = calCoeff(x, samples)
        coeffs.append(result)
    print(coeffs)
