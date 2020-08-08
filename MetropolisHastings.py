import sys
import os
import errorHandler
import argparse
import numpy as np
from math import *
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt


mean = np.array([5, 5])
cov = np.array([[0.5, 0.3],
                [0.3,0.7]])

def proposalDistribution(Beta, Gamma):
    # probability distribution function
    # normal distribution
    return [Beta, Gamma] + np.random.normal(size=2)


def pFunc(xStar, x):
    return st.multivariate_normal.pdf(xStar, x)

lambdaBeta= 0.5
nuBeta = 1
lambdaGamma= 0.5
nuGamma = 1

def gammaDistribution(Beta,Gamma):
    """
    param   Beta    : float
            Gamma   : float
    return pi(Beta, Gamma) with gamma distribution
    """
    # pi(Beta,Gamma) = pi(Beta)* pi(Gamma)
    # piBeta = ((nuBeta**lambdaBeta)*(Beta**(lambdaBeta-1))*(e**(-nuBeta*Beta)))/gamma(lambdaBeta)
    # piGamma = ((nuGamma**lambdaGamma)*(Gamma**(lambdaGamma-1))*(e**(-nuGamma*Gamma)))/gamma(lambdaGamma)
    
    piBeta = st.gamma.pdf(Beta,a=lambdaBeta,scale = 1/nuBeta)
    piGamma = st.gamma.pdf(Gamma,a=lambdaGamma,scale = 1/nuGamma)
    return piBeta*piGamma
def gaussDistribution(Beta,Gamma):
    """
    param   Beta    : float
            Gamma   : float
    return pi(Beta, Gamma) with gauss distribution
    """
    return st.multivariate_normal.pdf((Beta,Gamma), mean,cov)


def metropolisHastings(iter, piFunc, proposalDistribution = proposalDistribution, pFunc=None): 
    Beta0, Gamma0 = 1.,1.
    # Innitialize Beta0 and Gamma0 from fronterior probability distribution pi(Beta, Gamma).
    # by using Metropolis Hastings algorithm until the probability is large engough to accept.
    # The algorithm is run until we can find (Beta, Gamma) satisfied fronterior probability distribution pi(Beta, Gamma)
    while (piFunc(Beta0,Gamma0) < 5e-5):
        BetaStar, GammaStar = proposalDistribution(Beta0, Gamma0)
        # Since proposalDistribution is a symmetrical distribution, we can shortcut our calculation
        r = min(1, piFunc(BetaStar, GammaStar) / (piFunc(Beta0, Gamma0)))

        q = np.random.rand()
        if q < r:
            Beta0, Gamma0 = BetaStar, GammaStar
    samples = np.zeros((iter, 2))
    samples[0]= np.array([Beta0,Gamma0])
    # Start step 2
    for i in range(iter-1):
        # Assign Beta := Beta i and Gamma := Gamma i
        Beta, Gamma = samples[i]

        # Innitialize Beta* and Gamma* randomly from propability distribution p(Beta, Gamma).
        # Normal distribution is symmetrical
        # some symmetrical distribution: Cauchy distribution, logistic distribution, uniform distribution
        BetaStar, GammaStar = proposalDistribution(Beta, Gamma)

        #Since proposalDistribution is a symmetrical distribution, we can shortcut our calculation
        # r = min(1, piFunc(BetaStar, GammaStar) * pFunc((BetaStar, GammaStar), (Beta, Gamma)) / (piFunc(Beta, Gamma)*pFunc((Beta, Gamma), (BetaStar, GammaStar))))
        r = min(1, piFunc(BetaStar, GammaStar) / (piFunc(Beta, Gamma)))

        # Innitialize q from distribution U(0,1)
        q = np.random.rand()
        
        if q < r:
            Beta, Gamma = BetaStar, GammaStar
        # If q < r assign (Beta_i+1, Gamma_i+1) = (Beta*, Gamma*)
        # Else assign (Beta_i+1, Gamma_i+1) = (Beta, Gamma)
        samples[i+1] = np.array([Beta, Gamma])
    return samples


if __name__ == '__main__':
    result = metropolisHastings(10000, gaussDistribution)
    sns.jointplot(x=result[:, 0], y=result[:, 1])
    plt.show()
    plt.plot(result[:, 0], result[:, 1], linewidth=0.5)
    plt.show()
    print(result)
    plt.show()
