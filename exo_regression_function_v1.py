# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:12:02 2021

@author: MALIHA

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

def mean(dataset):
    result=sum(dataset)/len(dataset)         # compute mean
    return result

def variance(x):
    result = sum([pow(i-mean(x),2) for i in x])/(len(x)-1)  # variance
    return result

def covar(x, y):
    x_mean = mean(x)
    y_mean = mean(y)
    data = [(x[i] - x_mean) * (y[i] - y_mean)
            for i in range(len(x))]
    covar=sum(data)/(len(x)-1)
    return covar


def slope(x,y):
    slope=covar(x,y)/variance(x)  #slope compute
    return slope


def y_intercept(x,y):
    x_mean = mean(x)
    y_mean = mean(y)
    y_intercept=y_mean-slope(x,y)*x_mean  #intercept compute
    return y_intercept


def r2(x,y):
    r2=pow(covar(x,y),2)/(variance(x)*variance(y) )  #rsquare compute
    return r2

def std_error_cons(x,y):
    x_mean = mean(x)
    y_hat=slope(x,y)*x+y_intercept(x,y)
    p1=(1/len(x))
    p2=(pow(x_mean,2))
    p3=sum([pow(x[i]-x_mean,2) for i in range(len(x))])
    p4=p2/p3
    num=np.sqrt([pow(y[i] - y_hat,2) for i in range(len(y))])
    num1=np.sqrt(len(x)-2)
    s=num/num1
    stderr=s*(np.sqrt(p1+p4))
    return stderr
                                                                                
def std_error_slope(x,y):
    std_error=(slope(x,y)/(np.sqrt(len(x)-2)))*(np.sqrt(1/r2(x,y)-1))
    return std_error


def t_critical(alpha,df):
    t=stats.t.ppf(1-alpha/2,df)
    return t

def t_obs(x,y):
    t=slope(x,y)/std_error_slope(x,y)
    return t

def pvalue(x,y,df):
    pvalue = 2*(1 - stats.t.cdf(t_obs(x,y),df))
    return pvalue

def CI(x,y,alpha,df):
    
    #corresponding to α/2 (=0.025) and the line corresponding to n-p-1 degree of freedom (df=10-1-1=8)
    #Here the critical value (one tail test) is  t* (0.025,8)=2.306
    CIu=slope(x,y)+(t_critical(alpha,df)*std_error_slope(x,y))
    CIl=slope(x,y)-(t_critical(alpha,df)*std_error_slope(x,y))
    return CIu,CIl

def ctest(x,y,alpha,df):
    #hypothesis significance test for the slope
    #H0 slope is equal to zero
    #H1 slope is different to zero
    if  t_obs(x,y) >t_critical(alpha,df):
        print("reject the null hypothesis so slope is different from zero")  
    else:
        print("accept the null hypothesis so slope is equal to zero")
    return ctest

def func(x, m, b, c):
    return m*x+b

#%% MAIN CODE

x= [0,1,2,3,4,5,6,7,8,9]
y= [2,5,1,6,8,10,9,8,11,13]
alpha=0.05
p=1
df=len(x)-p-1
#popt, pcov = curve_fit(func, x, y)
#plt.plot(x, func(x, *popt), 'r-')


print("variance of x is",variance(x))
print("variance of y is",variance(y))
print("covariance is",{covar(x,y)})
print("slope is",slope(x,y))
print("intercept is",y_intercept(x,y))
print("r-square is",r2(x,y))
print("standard error of slope is",std_error_slope(x,y))
print("standard error of constant is",std_error_cons(x,y))
print("t critical is",t_critical(alpha,df))
print("t value is",t_obs(x,y))
print("p value is",pvalue(x,y,df))
print("CI IS",CI(x,y,alpha,df))
ctest(x,y,alpha,df)



x_con=x
x_con= sm.add_constant(x_con)
model = sm.OLS(y,x_con)
results = model.fit() 
results.params
print(results.summary())