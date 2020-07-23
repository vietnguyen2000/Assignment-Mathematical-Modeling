import sys
import os
import argparse
import numpy as np
import pandas as pd
from math import *
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

CONFIRMED = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
RECOVERED = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
DEATHS = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
PREDICT_RANGE=150

def predict_Euler_SIRD(beta, gamma, mu, S0, I0, R0, D0, predict_range):
    S = S0
    I = I0
    R = R0
    D = D0
    N= S+I+R+D
    listS, listI, listR, listD = [S], [I], [R], [D]
    for i in range(predict_range-1):
        dS = (-beta*I*S)/N
        dI = (beta*I*S/N - gamma*I - mu*I)
        dR = (gamma*I)
        dD = (mu*I)
        S = S + dS
        I = I + dI
        R = R + dR
        D = D + dD
        listS.append(S)
        listI.append(I)
        listR.append(R)
        listD.append(D)
    return np.array(listS),np.array(listI),np.array(listR),np.array(listD)

class Learner:
    def __init__(self, areaId, loss, start_date,N, beta = 0.001, gamma = 0.001, mu = 0.001):
        self.areaId = areaId
        self.area = self.getAreaName()

        self.loss = loss
        self.start_date = start_date

        self.beta = beta
        self.gamma= gamma
        self.mu = mu

        self.Rdata = self.load_recovered()
        self.Ddata = self.load_deaths()
        self.Idata = self.load_confirmed() - self.Rdata - self.Ddata
        Sdata = np.empty(self.Rdata.shape,dtype=np.int)
        Sdata.fill(N)
        self.Sdata = Sdata-self.Rdata-self.Idata-self.Ddata

        self.s_0 = self.Sdata[0]
        self.d_0 = self.Ddata[0]
        self.i_0 = self.Idata[0]
        self.r_0 = self.Rdata[0]
        
        self.predict_range = len(self.Idata) + PREDICT_RANGE
        self.index = self.extend_index(start_date, self.predict_range)
    
    def getAreaName(self):
        df = pd.read_csv(CONFIRMED)
        province, country = df.iloc[self.areaId, 0:2].tolist()
        if isnan(province): return country
        return province+ " "  + country


    def load_confirmed(self):
        df = pd.read_csv(CONFIRMED)
        dateIndex = (datetime.strptime(self.start_date, '%m/%d/%y') - datetime.strptime('1/22/20', '%m/%d/%y')).days +4
        confirm = np.array(df.iloc[self.areaId,dateIndex:].tolist())
        return confirm


    def load_recovered(self):
        df = pd.read_csv(RECOVERED)
        dateIndex = (datetime.strptime(self.start_date, '%m/%d/%y') - datetime.strptime('1/22/20', '%m/%d/%y')).days +4
        recovered = np.array(df.iloc[self.areaId,dateIndex:].tolist())
        return recovered
    
    def load_deaths(self):
        df = pd.read_csv(DEATHS)
        dateIndex = (datetime.strptime(self.start_date, '%m/%d/%y') - datetime.strptime('1/22/20', '%m/%d/%y')).days +4
        deaths = np.array(df.iloc[self.areaId,dateIndex:].tolist())
        return deaths

    def extend_index(self, start_date, new_size):
        current = datetime.strptime(start_date, '%m/%d/%y')
        values = [datetime.strftime(current, '%m/%d/%y')]
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values


    def graphic(self,beta,gamma,mu):
        Sdata = np.concatenate((self.Sdata,[None]*PREDICT_RANGE))
        Idata = np.concatenate((self.Idata,[None]*PREDICT_RANGE))
        Rdata = np.concatenate((self.Rdata,[None]*PREDICT_RANGE))
        Ddata = np.concatenate((self.Ddata,[None]*PREDICT_RANGE))
        predictS,predictI,predictR,predictD = predict_Euler_SIRD(beta,gamma,mu,self.s_0,self.i_0,self.r_0,self.d_0,self.predict_range)
        df = pd.DataFrame({"Susceptible data":Sdata,"Infected data":Idata,"Recovered data":Rdata,"Death data":Ddata,"Predict Susceptible": predictS, "Predict Infected": predictI, "Predict Recovered": predictR, "Predict Death": predictD},index = self.index)
        fig, ax = plt.subplots(figsize=(15, 10))
        df.plot(ax=ax)
        fig.savefig(f"{self.area}.png")

    def train(self):
        optimal = minimize(loss, [self.beta, self.gamma, self.mu], args=(self.Sdata, self.Idata, self.Rdata, self.Ddata , self.s_0, self.i_0, self.r_0, self.d_0), method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4), (0.00000001, 0.4)])
        print(optimal)
        beta, gamma, mu = optimal.x
        print(f"Area={self.area},beta={beta:.8f}, gamma={gamma:.8f}, mu={mu:.8f}, r_0:{(beta/(gamma+mu)):.8f}")
        return beta, gamma, mu
        
points = []
def loss(point, Sdata, Idata, Rdata, Ddata, S0, I0, R0, D0):
    global points
    size = len(Idata)
    beta, gamma, mu = point
    # print(point)
    points.append([beta,gamma,mu])
    listS, listI, listR, listD = predict_Euler_SIRD(beta, gamma, mu, S0, I0, R0, D0,size)
    
    maxI = np.amax(listI)
    maxAlpha = max(points[0])
    Ed1 = sum(np.square(np.log10(Idata)-np.log10(listI))+np.square(np.log10(Ddata)-np.log10(listD)))
    Ed2 = 0.01*np.log10(maxI)/maxI*sum(np.square(Idata-listI)+np.square(Ddata-listD))
    if(len(points)>2):
        sumEr = 0
        for i in range(1,len(points)-1):
            sumEr += (points[i][0]-points[i+1][0])**2 + (points[i][1]-points[i+1][1])**2 + (points[i][2]-points[i+1][2])**2

        Er = 100*log(maxI)/maxAlpha * sumEr
    else:
        Er = 0
    
    if(len(points)>1):
        E0 = 100*log(maxI)/maxAlpha * ((points[1][0]-points[0][0])**2 + (points[1][1]-points[0][1])**2 + (points[1][2]-points[0][2])**2)
    else:
        E0 = 0

    return Ed1 + Ed2 + Er + E0


learner = Learner(6,loss,"3/14/20",500000,0.05,0.001,0.0001)
beta, gamma, mu = learner.train()
learner.graphic(beta,gamma,mu)

