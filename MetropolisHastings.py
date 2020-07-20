import sys
import os
import errorHandler
import argparse
import numpy as np
from math import *
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt


mean = np.array([0.5, 0.5])
cov = np.array([[0.5, 0.],
                [0.,0.5]])

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
    return π(β, γ) with gamma distribution
    """
    # π(β,γ) = π(β)* π(γ)oj
    # piBeta = ((nuBeta**lambdaBeta)*(Beta**(lambdaBeta-1))*(e**(-nuBeta*Beta)))/gamma(lambdaBeta)
    # piGamma = ((nuGamma**lambdaGamma)*(Gamma**(lambdaGamma-1))*(e**(-nuGamma*Gamma)))/gamma(lambdaGamma)
    piBeta = st.gamma.pdf(Beta,a=lambdaBeta,scale = 1/nuBeta)
    piGamma = st.gamma.pdf(Gamma,a=lambdaGamma,scale = 1/nuGamma)
    return piBeta*piGamma
def gaussDistribution(Beta,Gamma):
    """
    param   Beta    : float
            Gamma   : float
    return π(β, γ) with gauss distribution
    """
    return st.multivariate_normal.pdf((Beta,Gamma), mean,cov)


def metropolisHastings(iter, piFunc, proposalDistribution = proposalDistribution, pFunc=None):
    Beta0, Gamma0 = 1.,1.
    # Khởi tạo β0 và γ0 từ phân bố xác suất tiên nghiệm π(β, γ).
    # bằng cách sử dụng giải thuật Metropolis Hastings cho đến khi xác suất đủ lớn để chấp nhận.
    # hay còn gọi là chạy giải thuật cho đến khi tìm được một cặp (β, γ) thỏa phân bố xác suất tiên nghiệm π(β, γ)
    while (piFunc(Beta0,Gamma0) < 5e-5):
        BetaStar, GammaStar = proposalDistribution(Beta0, Gamma0)
        # vì proposalDistribution là phân phối xác suất đối xứng vì thế ta rút gọn tính toán
        r = min(1, piFunc(BetaStar, GammaStar) / (piFunc(Beta0, Gamma0)))

        q = np.random.rand()
        if q < r:
            Beta0, Gamma0 = BetaStar, GammaStar
    samples = np.zeros((iter, 2))
    samples[0]= np.array([Beta0,Gamma0])
    # bắt đầu bước 2
    for i in range(iter-1):
        # gán β := βi và γ := γi
        Beta, Gamma = samples[i]

        # Khởi tạo β* và γ* ngẫu nhiên từ phân phối xác suất bất kỳ p(β, γ).
        # Phân phối chuẩn là phân phối có tính chất đối xứng
        # Một vài phân phối khác cũng có tính đối xứng: Cauchy distribution, logistic distribution, uniform distribution
        BetaStar, GammaStar = proposalDistribution(Beta, Gamma)

        # vì proposalDistribution là phân phối xác suất đối xứng vì thế ta rút gọn tính toán
        # r = min(1, piFunc(BetaStar, GammaStar) * pFunc((BetaStar, GammaStar), (Beta, Gamma)) / (piFunc(Beta, Gamma)*pFunc((Beta, Gamma), (BetaStar, GammaStar))))
        r = min(1, piFunc(BetaStar, GammaStar) / (piFunc(Beta, Gamma)))

        # khởi tạo q từ phân phối đều liên tục U(0,1)
        q = np.random.rand()
        
        if q < r:
            Beta, Gamma = BetaStar, GammaStar
        # nếu q < r thì gán (β_i+1, γ_i+1) = (β*, γ*)
        # ngược lại thì gán (β_i+1, γ_i+1) = (β, γ)
        samples[i+1] = np.array([Beta, Gamma])
    return samples


if __name__ == '__main__':
    result = metropolisHastings(10000, gaussDistribution)
    sns.jointplot(x=result[:, 0], y=result[:, 1])
    print(result)
    plt.show()
