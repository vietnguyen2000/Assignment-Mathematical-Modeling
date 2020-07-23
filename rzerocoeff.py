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
    pi = st.gamma.pdf(x,a=Beta,scale=1/Gamma)
    return np.prod(pi)
    # for i in range(1,weeks):
    #     # if i ==0 : Xi = X[0]
    #     Xi = X[i*7] - X[(i-1)*7]
    #     # print(Xi)
    #     # print ([i,X[i],X[i-1],Xi])
    #     # try:
    #     if Xi > 0:
    #         # print(Gamma, Beta)
    #         # product *= (pow(Gamma, Beta)/gamma(Beta) * pow(Xi, (Beta - 1)) * pow(e, (-Gamma * Xi)))
    #         product *= st.gamma.pdf(Xi,a=Beta,scale= 1/Gamma)
    #         # pi.append(st.gamma.pdf(Xi,a=Beta,scale= 1/Gamma))
    #     else:
    #         return 0
    #     #     product = 0
    #     #     break
    #     # print(pow(e, -Gamma * Xi))
    #     # if isnan(product):
    #     #     print("sdoishdsisihdsdi ", gamma(Beta), e ** (-Gamma * Xi), ' ', Gamma * Xi)
    #     #     # break
    #     #     product = 0
    #     # product *= st.gamma.pdf(Xi,a=Beta,scale= 1/Gamma)
    #         # print(product)
    #     # except ValueError:
    #     #     # pass
    #     #     print(Gamma, Beta, Xi)
    #     # except OverflowError:
    #     #     print(pow(e, (-Gamma * Xi)))
    # # product *= ((Gamma ** Beta)/gamma(Beta) ** len(X))
    # # print(product)
    
    # return product

# Hàm hiện thực công thức 21:
def calCoeff(X, samples):
    rzero = 0
    # normalizedSamples = np.zeros(samples.shape)
    # normalizedSamples[:,0] = (samples[:,0] - np.min(samples[:,0]))/np.ptp(samples[:,0])
    # normalizedSamples[:,0] = samples[:,0]
    # normalizedSamples[:,1] = (samples[:,1] - np.min(samples[:,1]))/np.ptp(samples[:,1])
    # print(normalizedSamples)
    # print(rzero)
    
    # print(x)
    pi = []
    for sample in samples:
        pi.append(calPosterior(x, sample[0], sample[1]))
    Pdata =sum(pi)
    print("done pi")
    n=0
    for sample in samples:
        # print(sample)
        if sample[1] != 0 and sample[0] != 0 and pi[n]>0:
            rzero += (pi[n] * sample[0]/sample[1])/Pdata
        # print(rzero)
        n+=1
    return rzero

if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    url2 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    url3 = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    df = pd.read_csv(url, error_bad_lines=False)
    df2 = pd.read_csv(url2, error_bad_lines=False)
    df3 = pd.read_csv(url3, error_bad_lines=False)
    # Chọn các quốc gia: Germany (122), France (118), United Kingdom (225)
    indexes = 228
    coeffs = list()
    df = df.iloc[indexes]
    df2 = df2.iloc[indexes]
    df3 = df3.iloc[indexes]
    # print(df.isna())
    # for i in range(len(indexes)):
        # Đi từ index 5 tương ứng ngày đầu tiên 01/22/20
    X = df[4:].tolist()
    X2 = df2[4:].tolist()
    X3 = df3[4:].tolist()
    x = []
    for i in range(len(X)):
        if i == 0:
            s = (X[i] + X2[i] +X3[i])
        s = (X[i]-X[i-1] -X2[i]+X2[i-1] +X3[i]-X3[i-1])
        if s >0:
            x.append(s)
    # X_norm = [float(i)/500000 for i in X]
    print(x)
    samples = metropolisHastings(10000, gammaDistribution)
    print("done samples")
    # print(samples)
    result = calCoeff(x, samples)
    print(result)

    #     coeffs.append(result)
    # print(coeffs)