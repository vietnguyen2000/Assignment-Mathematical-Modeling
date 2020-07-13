import sys
import os
import errorHandler
import argparse
import numpy as np
from math import *
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt


def readCommand():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--testcase',
                        required=True,
                        default=None,
                        help='You just type the filename in testcase/Euler/ directory. This option is requirement.')
    parser.add_argument('-s', '--sample',
                        type=int,
                        required=False,
                        dest='samples',
                        default=10000,
                        help='The samples you want to run your algorithm. Default is 10000.')
    return parser
def calculateDistribution(variable, function):
    e = np.e
    pi = np.pi
    Beta = variable[0]
    Gamma = variable[1]
    return eval(function)

def metropolis_hastings(function, iter):
    Beta, Gamma = 0.,0.
    samples = np.zeros((iter, 2))

    for i in range(iter):
        # Phân phối chuẩn là phân phối có tính chất đối xứng
        # Một vài phân phối khác cũng có tính đối xứng: Cauchy distribution, logistic distribution, uniform distribution
        BetaStar, GammaStar = [Beta,Gamma] + np.random.normal(size=2)
        q = np.random.rand()
        r = min(1,calculateDistribution((BetaStar, GammaStar),function) / calculateDistribution((Beta, Gamma),function))
        if q < r:
            Beta, Gamma = BetaStar, GammaStar
        samples[i] = np.array([Beta, Gamma])
    return samples
    
def readDistribution(filename):
    f = open(filename, 'r')
    function = f.readline()
    return function

if __name__ == '__main__':
    try:
        options = readCommand().parse_args()
    except:
        readCommand().print_help()
        sys.exit(0)
    function = readDistribution("./testcase/MetropolisHastings/" + options.testcase)
    result = metropolis_hastings(function,options.samples)
    sns.jointplot(x=result[:, 0], y=result[:, 1])
    print(result)
    plt.show()
    
