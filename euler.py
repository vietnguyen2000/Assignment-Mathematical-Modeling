import optparse
import errorHandler
import os
import sys
def readCommand(argv):
    parser = optparse.OptionParser()
    parser.add_option('-m','--model',
                      dest='model',
                      help='Model SIR or SIRD. this option is requirement.')
    parser.add_option('-t','--testcase',
                      dest='testcase',
                      help='You just type the filename in testcase/Euler/ directory. This option is requirement.')
    parser.add_option('-s','--step',
                      type='int',
                      dest='step',
                      default=8,
                      help='The steps you want to run your algorithm. Default is 8')                  
    (options, _) = parser.parse_args(argv)
    return options

def Euler_SIR(filename,step):
    testcase = _readEulerTestcase(filename)
    t, B, g, S, I, R = testcase
    result = []
    result.append((S,I,R))
    for i in range(step):
        dS = -B*I*S
        dI = B*I*S - g*I
        dR = g*I
        S = S + dS
        I = I + dI
        R = R + dR
        result.append((S,I,R))
    _printEulerSIRData(result,t)
    return result
def _printEulerSIRData(data,deltaT = 1):
    print("Ngày\tCó nguy cơ\tCa nhiễm\tCa hồi phục")
    for i in range(len(data)):
        print(int(i*deltaT),"\t %0.5f"%data[i][0],"\t %0.5f"% data[i][1],"\t %0.5f"%data[i][2])

def Euler_SIRD(filename,step):
    testcase = _readEulerTestcase(filename)
    t, B, g, u, S, I, R, D= testcase
    result = []
    result.append((S,I,R,D))
    for i in range(step):
        dS = -B*I*S
        dI = B*I*S - g*I -u*I
        dR = g*I
        dD = u*I
        S = S + dS
        I = I + dI
        R = R + dR
        D = D + dD
        result.append((S,I,R,D))
    _printEulerSIRDData(result,t)
    return result
def _printEulerSIRDData(data,deltaT = 1):
    print("Ngày\tCó nguy cơ\tCa nhiễm\tCa hồi phục\tCa chết")
    for i in range(len(data)):
        print(int(i*deltaT),"\t %0.5f"%data[i][0],"\t %0.5f"% data[i][1],"\t %0.5f"%data[i][2],"\t %0.5f"%data[i][3])

def _readEulerTestcase(filename):
    f=open(filename,'r')
    result = []
    rows = []
    for line in f:
        row = line.split()
        result.append(float(row[1]))
    return result

if __name__ == '__main__':
    options = readCommand(sys.argv)
    if(options.model == "SIR"):
        if(options.testcase == None):
            Euler_SIR("./testcase/Euler/SIR.txt",options.step)
        else:
            Euler_SIR("./testcase/Euler/"+options.testcase,options.step)
    elif(options.model == "SIRD"):
        if(options.testcase == None):
            Euler_SIRD("./testcase/Euler/SIRD.txt",options.step)
        else:
            Euler_SIRD("./testcase/Euler/"+options.testcase,options.step)
    else:
        errorHandler.commandMissing("--model")