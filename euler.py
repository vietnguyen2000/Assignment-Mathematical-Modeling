import sys
import os
import errorHandler
import argparse


def readCommand():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        required=True,
                        default=None,
                        help='Model SIR or SIRD. this option is requirement.')
    parser.add_argument('-t', '--testcase',
                        required=True,
                        default=None,
                        help='You just type the filename in testcase/Euler/ directory. This option is requirement.')
    parser.add_argument('-s', '--step',
                        type=int,
                        required=False,
                        default=8,
                        help='The steps you want to run your algorithm. Default is 8. Steps is the number of interval time from the beginning time to the end time')
    return parser


def Euler_SIR(filename, step):
    testcase = _readEulerTestcase(filename)
    dentaT, B, g, S, I, R = testcase
    result = []
    result.append((0, S, I, R))
    for i in range(step):
        dS = (-B*I*S) * dentaT
        dI = (B*I*S - g*I) * dentaT
        dR = (g*I) * dentaT
        S = S + dS
        I = I + dI
        R = R + dR
        result.append((1+i*dentaT, S, I, R))
    _printEulerSIRData(result)
    return result


def _printEulerSIRData(data):
    print("Tuần\tCó nguy cơ\tCa nhiễm\tCa hồi phục")
    for i in range(len(data)):
        print(int(data[i][0]), "\t %0.5f" %
              data[i][1], "\t %0.5f" % data[i][2], "\t %0.5f" % data[i][3])


def Euler_SIRD(filename, step):
    testcase = _readEulerTestcase(filename)
    dentaT, B, g, u, S, I, R, D = testcase
    result = []
    result.append((0, S, I, R, D))
    for i in range(step):
        dS = (-B*I*S) * dentaT
        dI = (B*I*S - g*I - u*I) * dentaT
        dR = (g*I)*dentaT
        dD = (u*I)*dentaT
        S = S + dS
        I = I + dI
        R = R + dR
        D = D + dD
        result.append((1+i*dentaT, S, I, R, D))
    _printEulerSIRDData(result)
    return result


def _printEulerSIRDData(data, deltaT=1):
    print("Tuần\tCó nguy cơ\tCa nhiễm\tCa hồi phục\tCa chết")
    for i in range(len(data)):
        print(data[i][0], "\t %0.5f" % data[i][1], "\t %0.5f" %
              data[i][2], "\t %0.5f" % data[i][3], "\t %0.5f" % data[i][4])


def _readEulerTestcase(filename):
    f = open(filename, 'r')
    result = []
    rows = []
    for line in f:
        row = line.split()
        result.append(float(row[1]))
    return result


if __name__ == '__main__':
    try:
        options = readCommand().parse_args()
    except:
        readCommand().print_help()
        sys.exit(0)
    if options.model == 'SIR':
        Euler_SIR('./testcase/Euler/'+options.testcase,
                  options.step)
    else:
        Euler_SIRD('./testcase/Euler/'+options.testcase,
                   options.step)
