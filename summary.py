# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:41:17 2021

@author: MALIHA
"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Reading the data
data=np.genfromtxt("training_timeseries.csv", skip_header=1,delimiter=',')

#Plotting the data
fig=plt.figure()
ax=fig
plt.plot(data[:,0],data[:,1])
plt.xlabel('year')
plt.ylabel('water level (m)')
plt.show()
#5 years water level variation in harmonic way

#Defing function for curve fit
def f(t,slope,intercept,amplitude,delta):
    return slope*t+intercept+amplitude*np.cos((2*np.pi/1)*t-delta)
    

param, param_cov=curve_fit(f,data[:,0],data[:,1])

print("Cosine funcion coefficients:")
print(param)
print("Covariance of coefficients:")
print(param_cov)

ans=(param[0]*data[:,0])+param[1]+param[2]*(np.cos((2*np.pi/1)*data[:,0])-param[3])

#Plotting
plt.figure()
plt.plot(data[:,0],data[:,1], '*', color ='blue', label ="data")
plt.plot(data[:,0],ans, '-', color ='red', label ="optimized data")
plt.legend()
plt.show()