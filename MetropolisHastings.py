import sys
import os
import errorHandler
import argparse
import numpy as np
from math import *
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt


def proposal_distribution_func(Beta, Gamma):
    # probability distribution function
    # normal distribution
    return [Beta, Gamma] + np.random.normal(size=2)


def pFunc(xStar, x):
    return st.multivariate_normal.pdf(xStar, x)


def piFunc(x):
    # probability density function
    return st.multivariate_normal.pdf(x, [1.4, 0.4])


def metropolis_hastings(iter, piFunc, pFunc, init):
    Beta, Gamma = init
    samples = np.zeros((iter, 2))
    for i in range(iter):
        # Phân phối chuẩn là phân phối có tính chất đối xứng
        # Một vài phân phối khác cũng có tính đối xứng: Cauchy distribution, logistic distribution, uniform distribution
        BetaStar, GammaStar = proposal_distribution_func(Beta, Gamma)
        q = np.random.rand()
        r = min(1, piFunc((BetaStar, GammaStar)) * pFunc((BetaStar, GammaStar), (Beta,
                                                                                 Gamma)) / (piFunc((Beta, Gamma))*pFunc((Beta, Gamma), (BetaStar, GammaStar))))
        if q < r:
            Beta, Gamma = BetaStar, GammaStar
        samples[i] = np.array([Beta, Gamma])
    return samples


if __name__ == '__main__':
    init = np.array([3, 4])
    result = metropolis_hastings(10000, piFunc, pFunc, init)
    sns.jointplot(x=result[:, 0], y=result[:, 1])
    print(result)
    plt.show()
