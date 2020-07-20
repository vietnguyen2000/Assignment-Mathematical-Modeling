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
def calPosterior(X, Beta, Gamma):
    product = 1.
    for i in range(len(X)):
        if i ==0 : Xi = X[0]
        else: Xi = X[i]-X[i-1]
        try:
            if Xi != 0:
                # print(Gamma, Beta)
                # product *= (pow(Gamma, Beta)/gamma(Beta) * pow(Xi, (Beta - 1)) * pow(e, (-Gamma * Xi)))
                product *= st.gamma.pdf(Xi,a=Gamma,scale= 1/Beta)
        # else:
        #     product = 0
        #     break
        # print(pow(e, -Gamma * Xi))
        # if isnan(product):
        #     print("sdoishdsisihdsdi ", gamma(Beta), e ** (-Gamma * Xi), ' ', Gamma * Xi)
        #     # break
        #     product = 0
        # product *= st.gamma.pdf(Xi,a=Beta,scale= 1/Gamma)
            # print(product)
        except ValueError:
            # pass
            print(Gamma, Beta, Xi)
        except OverflowError:
            print(pow(e, (-Gamma * Xi)))
    # product *= ((Gamma ** Beta)/gamma(Beta) ** len(X))
    # print(product)
    return product

# Hàm hiện thực công thức 21:
def calCoeff(X, samples):
    rzero = 0
    normalizedSamples = np.zeros(samples.shape)
    normalizedSamples[:,0] = (samples[:,0] - np.min(samples[:,0]))/np.ptp(samples[:,0])
    normalizedSamples[:,1] = (samples[:,1] - np.min(samples[:,1]))/np.ptp(samples[:,1])
    # print(normalizedSamples)
    # print(rzero)
    for sample in normalizedSamples:
        # print(sample)
        if sample[1] != 0 and sample[0] != 0:
            rzero += (calPosterior(X, sample[0], sample[1]) * sample[0]/sample[1])
        # print(rzero)
    return rzero

if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    df = pd.read_csv(url, error_bad_lines=False)
    # Chọn các quốc gia: Germany (122), France (118), United Kingdom (225)
    indexes = [116, 120, 223]
    coeffs = list()
    df = df.iloc[indexes,0:200]
    # print(df.isna())
    for iter in df.iterrows():
        # Đi từ index 5 tương ứng ngày đầu tiên 01/22/20
        X = iter[1][5:].tolist()
        # print(X)
        # X_norm = [float(i)/500000 for i in X]
        # print(X)

        samples = metropolisHastings(10000, gammaDistribution)

        # print(samples)
        coeffs.append(calCoeff(X, samples))
    print(coeffs)