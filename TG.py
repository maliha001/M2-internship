# -*- coding: utf-8 -*- 

"""
Created on Mon May 17 09:32:20 2021

@author: MALIHA
"""

#%%
import pandas as pd
from datetime import date,timedelta,datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
import numpy as np 
import seaborn as sns

##%

def date_converter(y,m,d,h):
    my_string=str(y)+'-'+str(m).zfill(2)+'-'+str(d).zfill(2)+' '+str(h).zfill(2)
    return pd.to_datetime(my_string,format='%Y-%m-%d %H')

def uhslc_csv_reader(file):
    print(f'...reading {file}')
    df=pd.read_csv(file,names=["year","months","day","hour",'value'],delimiter=",",
               parse_dates={'datetime':['year','months','day','hour']},date_parser=date_converter,
              na_values=-32767)
    df.set_index('datetime',inplace=True)
    return df

def date_converter1(year):
    return pd.Timestamp(str(year))

def psml_csv_reader(file):
    df=pd.read_csv(file, 
                   delimiter=';',
                   names=["year", "values", "flags", "null"],
                   na_values=-99999,
                   parse_dates={"datetime":["year"]},
                   date_parser=date_converter1)
    df.drop(columns=['flags', 'null'], inplace=True)
    df.set_index("datetime", inplace=True)
    return df

def trend_ols(y,x,summary=False):
    ''' '''
    model=sm.OLS(y,sm.add_constant(x),missing='drop')
    results=model.fit()
    intercept,slope = results.params
    std_error=results.bse[1]
    y_fit=  slope*x+intercept
    if summary :
        print(results.summary())
    else :
        print(f" (trend_ols) slope is {slope:.3f} mm/yr +/- {results.bse[1]:.3f}")
    return slope,intercept,std_error

def f(x,slope,intercept,amplitude,delta):
    T=365 #time period for annual 
    x2=x-x[0]
    return slope*x2+intercept+amplitude*np.cos((2*np.pi/T)*x2-delta)

def fit_function(df,f):
    " fit the function f on the dataframe df"
    param,param_cov=curve_fit(f,df.index.values,
                              df.values,
                              #bounds=[(-1.4e-4,-1,0,0),(1.4e-4,1,5,2*np.pi)])
                              bounds=[(-np.inf,-np.inf,0,0),(np.inf,np.inf,np.inf,2*np.pi)])
    stdev = np.sqrt(np.diag(param_cov))
    print(f" (fit_function) 1.slope {param[0]*1000.*365:.3f} mm/yr +/- {stdev[0]*1000.*365:.3f}")
    print(f" (fit_function) 2.intercept {param[1]:.3f} m +/- {stdev[1]:.3f}")
    print(f" (fit_function) 3.amplitude {param[2]*10:.3f} cm +/- {stdev[2]:.3f}")
    print(f" (fit_function) 4.phase {param[3]:.3f} radian +/- {stdev[3]:.3f}")
    return param[0]*df.index.values+param[1]+param[2]*np.cos((2*np.pi/365)*df.index.values-param[3])

##%
# reqding the raw data
file=r"C:\Users\MALIHA\Documents\Project/Cox1.csv"
file1=r"D:\classes\PSML\ctg.txt"
df=uhslc_csv_reader(file)
dfp=psml_csv_reader(file1)
dfps=dfp.dropna()

##UHCL TG
ax=df.plot(label='raw data')
df.resample('M').mean().plot(ax=ax,label='monthly mean')
plt.title(' observation data of tide gauge')
plt.ylabel('sea-level (mm)')
plt.xlabel('Year')
ax.legend()

year=df.index.values.astype(float)
df['year']=year
df1=df.loc['1983':'2000']
dft=df1.dropna()

ols=trend_ols(df['value'],df.index.values.astype(float),summary=True)
slope,intercept,std_error=trend_ols(df1['value'],df1.index.values.astype(float),summary=False)
y_fit=slope*df1.index.values.astype(float)+intercept

fig,ax=plt.subplots()
figsize=(20,20)
ax.scatter(df1.index,df1['value'],color='blue',marker='o',label='observed data')
ax.plot(df1.index,y_fit,'green',linewidth=5,label='trend line')
plt.legend()

param,param_cov=curve_fit(f,dft.index.values.astype(float)/1e9/3600/24,dft.value.values, bounds=[(-np.inf,-np.inf,0,0),(np.inf,np.inf,np.inf,2*np.pi)])
stdev = np.sqrt(np.diag(param_cov))
fitted = param[0]*dft.index.values.astype(float)+param[1]+param[2]*np.cos((2*np.pi/365)*dft.index.values.astype(float)-param[3])
residual=dft.value.values-fitted
#bounds=[(-1.4e-4,-1,0,0),(1.4e-4,1,5,2*np.pi)])
                             
print(f" (fit_function) 1.slope {param[0]*1000.*365:.3f} mm/yr +/- {stdev[0]*1000.*365:.3f}")
print(f" (fit_function) 2.intercept {param[1]:.3f} m +/- {stdev[1]:.3f}")
print(f" (fit_function) 3.amplitude {param[2]*10:.3f} cm +/- {stdev[2]:.3f}")
print(f" (fit_function) 4.phase {param[3]:.3f} radian +/- {stdev[3]:.3f}")

#plot model
fig = plt.figure(figsize=(10, 4))
plt.plot(dft.index,dft.value.values, '*', color ='blue', label ="data")
plt.plot(dft.index,fitted, '-', color ='red', label ="model fit")
plt.plot(dft.index, residual, 'o', color='green', label="residual")
plt.xlabel('year')
plt.ylabel('Sea level anomaly(m)')
plt.legend(loc='lower right')
plt.title(f'slope {param[0]*1000.*365:.1f} mm/yr +/- {stdev[0]*1000.*365:.1f}\namplitude {param[2]*100:.2f} cm +/- {stdev[2]*100:.2f}; phase {np.degrees(param[3]):.1f} degrees +/- {np.degrees(stdev[3]):.1f}')
plt.show()

##PSML TG
ax=dfps.plot(label='raw data')
plt.title(' observation data of tide gauge')
plt.ylabel('sea-level (mm)')
plt.xlabel('Year')
ax.legend()
#curve fit
b=dfps.index.values.astype(float)/1e9/3600/24
param1,param_cov1=curve_fit(f,b,dfps.values,bounds=[(-np.inf,-np.inf,0,0),(np.inf,np.inf,np.inf,2*np.pi)])
stdev = np.sqrt(np.diag(param_cov1))
fitted = param1[0]*dfps.index.values.astype(float)+param1[1]+param1[2]*np.cos((2*np.pi/365)*dfps.index.values.astype(float)-param1[3])
residual=dfps.value.values-fitted

