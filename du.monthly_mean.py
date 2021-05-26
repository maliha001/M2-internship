# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:55:50 2021

@author: MALIHA
"""
#%%
import xarray as xr 
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import statsmodels.api as sm

#%%

def f(t,slope,intercept,amplitude,delta, T=365.25):
    return slope*t+intercept+amplitude*np.cos((2*np.pi/T)*t-delta)
#%%

ds = xr.open_dataset('duacs_monthly.nc')
ds
df=ds.sla.sel(latitude=21.58,longitude=91.12,method='nearest').to_dataframe()
df
param, param_cov=curve_fit(f,df.index.values.astype(float)/1e9/3600/24,df.sla.values,bounds=[(-np.inf,-np.inf,0,0),(np.inf,np.inf,np.inf,2*np.pi)])
fitted = param[0]*df.index.values.astype(float)/1e9/3600/24+param[1]+param[2]*np.cos((2*np.pi/365)*df.index.values.astype(float)/1e9/3600/24-param[3])
residual=df.sla.values-fitted
stdev = np.sqrt(np.diag(param_cov))

monthly_mean = ds['sla'].groupby('time.month').mean()

#plt.figure()
#monthly_mean.plot()
monthly_mean
monthly_amp = (monthly_mean.max(dim='month')-monthly_mean.min(dim='month'))/2
m=monthly_amp.sel(latitude=21.58,longitude=91.12,method='nearest')
AOI_amp=m.values*100 
#%%
#plot model
fig = plt.figure(figsize=(10, 4))
plt.plot(df.index,df.sla.values, '*', color ='blue', label ="data")
plt.plot(df.index,fitted, '-', color ='red', label ="model fit")
plt.plot(df.index.values, residual, 'o', color='green', label="residual")
plt.xlabel('year')
plt.ylabel('Sea level anomaly(m)')
plt.legend(loc='lower right')
plt.title(f'slope {param[0]*1000.*365:.1f} mm/yr +/- {stdev[0]*1000.*365:.1f}\namplitude {param[2]*100:.2f} cm +/- {stdev[2]*100:.2f}; phase {np.degrees(param[3]):.1f} degrees +/- {np.degrees(stdev[3]):.1f}')
plt.show()
#%%
'''plotting histogram'''
# fig, axes = plt.subplots(ncols=2, figsize=(8, 4), sharex=True)
# axes[0].hist(df.sla.values, bins='fd')
# axes[0].set_title(f'SLA, std={np.std(df.sla.values, ddof=1):0.3f}')
# axes[1].hist(residual, bins='fd')
# axes[1].set_title(f'Residual, std={np.std(residual, ddof=1):0.3f}')
# plt.show()


ax=monthly_amp.plot(label=False)
plt.gca().axes.get_cbar().set_visible(False)
plt.gca.legend_ =None
ax.get_legend().remove()
cbar = plt.colorbar()
cbar.set_label('Amplitude of SLA', rotation=270)
ax.coastlines(resolution='10m',zorder=2)



ax.add_feature(cfeature.BORDERS,zorder=1)
ax.add_feature(cfeature.RIVERS,facecolor='blue',zorder=3)
ax.gridlines(draw_labels=True,zorder=5)
ax.add_feature(cfeature.LAND,facecolor="grey")