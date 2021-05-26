
# coding: utf-8

# import of the module
from pathlib import Path
import cartopy.crs as ccrs
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy import stats
import statsmodels.api as sm
import datetime 
import math
import statistics
import statsmodels.api as sm
from datetime import datetime
from scipy.optimize import curve_fit

plt.close('all')

# load the dataset
file=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-053-fv01.1.nc")
ds=xr.open_dataset(file,decode_times=False)


#%% define my function

def select_point(ds,point,verbose=True):
    ''' select a point in a track and return a dict with needed parameters'''
    #ref=pd.Timestamp(1950,1,1)
    print(f"extract point #{point} on track {ds.pass_number}")
    output={}
    output["dist"]=ds.dist_to_coast_gshhs[point].values/1000.
    output['pt']=point
    output["location"]=[ds.lon[point].values,ds.lat[point].values]
    #time = pd.to_datetime(ds.time[point,:], unit='D', origin=ref)
    time = ds.time[point,:].to_pandas() # To panda dataframe
    sla  = ds.sla[point,:].squeeze() # Squeeze the unused dim if any
    output["ts"]=pd.DataFrame(index=time,data={'sla':sla})
    output["valid"] = (1-(output['ts'].isna().sum().values[0] / len(output['ts'])))*100.
    output["trend"],intercept,std_error= trend_ols(sla.values,ds.time[point,:].values)
    output['trend_rate_with_outlier']=output["trend"]*1000*365
    output['flagged']=sigma_mask(output['ts'])
    return output

def sigma_mask(df,n=2):
    '''''''removing outliers'''
    sigma = df.std()
    mask = (df >- n*sigma) & (df < n*sigma)
    df_masked=df[mask.values]
    return df_masked

def flagged_trend(ds,point):
    
    time = ds.time[point,:]
    sla  = ds.sla[point,:]
    df=pd.DataFrame(index=time,data={ds['sla'].attrs['long_name']:sla})
    df_masked=sigma_mask(df,n=2)
    slope,intercept,std_error=trend_ols(df_masked.values,df_masked.index.values)
    trend_rate=slope*1000*365
    print ('trend rate is :',(trend_rate),'mm/yr' )

def plot_point(ds,point):
    '''ploting points without outliers'''
    ref=pd.Timestamp(1950,1,1)
    output=select_point(ds,point)
    title=f"Point No: {point} - Distance to Coast: {output['dist']:.2f} km-{output['valid']:.2f}%"
    df=output['ts']
    df_flagged=output['flagged']
    time=pd.to_datetime(output['ts'].index, unit='D', origin=ref)
    time_flagged=pd.to_datetime(output['flagged'].index, unit='D', origin=ref)

    #plots
    fig,ax=plt.subplots()
    ax.scatter(time,df.values,color='red')
    ax.scatter(time_flagged,df_flagged.values,color='skyblue')
    slope,intercept,std_error=trend_ols(df_flagged.values,df_flagged.index.values)
    s_mm=slope*1000*365
    s_mm="%.2f" % s_mm
    e_mm=std_error*1000*365
    e_mm="%.2f" % e_mm
    label=f"{s_mm} mm/yr +/- {e_mm} mm/yr"
    y_fit=  slope*df_flagged.index.values+intercept
    ax.plot(time_flagged,y_fit,linewidth=2.,c='green',label=label)
    ax.set_yticks(np.arange(-0.5, 1, 0.5))
    ax.set_title(title)
    ax.legend(loc=1)
    plt.ylabel("Sea Level Anomaly (m)")
    plt.grid()
    return ax

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
        print(f" (trend_ols) 1.slope is {slope*1000.*365.:.3f} mm/yr +/- {results.bse[1]*1000.*365.:.3f}")
    return slope,intercept,std_error

def map_selected_point(ds,point,extent=None):
    dl=1.
    if extent is None :
        extent = [ds.lon.min()-dl, ds.lon.max()+dl, ds.lat.min()-dl, ds.lat.max()+dl]
    fig, ax = plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()})
    ax.set_extent(extent)
    ax.coastlines(resolution='10m') 
    ax.gridlines(draw_labels=True)
    ax.stock_img()
    ax.scatter(ds.lon.values,ds.lat.values,color='black',marker='.', s=0.01)
    ax.scatter(ds.lon[point].values,ds.lat[point].values,color='red',marker='o',zorder=1)
    return ax

#Defing function for curve fit
def f(t,slope,intercept,amplitude,delta, T=365.25):
    return slope*t+intercept+amplitude*np.cos((2*np.pi/T)*t-delta)

def plot_model(ds, point):
    ""
    ref=pd.Timestamp(1950,1,1)
    output=select_point(ds,point)
    title=f"Point No: {point} - Distance to Coast: {output['dist']:.2f} km-{output['valid']:.2f}%"
    df=output['ts']
    df_flagged=output['flagged']
    time=pd.to_datetime(output['ts'].index, unit='D', origin=ref)
    time_flagged=pd.to_datetime(output['flagged'].index, unit='D', origin=ref)
    fitted_model=fit_function(df_flagged, f)
    residual=df_flagged.values.flatten()-fitted_model
     #Plotting
    fig = plt.figure(figsize=(10, 4))
    plt.plot(time_flagged,df_flagged.values, '*', color ='blue', label ="data")
    plt.plot(time_flagged,fitted_model, '-', color ='red', label ="model fit")
    plt.plot(time_flagged,residual, 'o', color ='green', label ="residual")
    plt.xlabel('year')
    plt.ylabel('Sea level anomaly(m)')
    plt.legend(loc='lower right')
    plt.show()

def fit_function(df, f, T=365.25):
    '''
    fit the function f on the dataframe df
    '''
    t = df.index.values.flatten()
    t_tr = t - t[0]
    values = df.values.flatten()
    param, param_cov=curve_fit(
        f,
        t_tr,
        values,
        #bounds=[(-1.4e-4,-1,0,0),(1.4e-4,1,5,2*np.pi)])
        bounds=[(-np.inf,-np.inf,0,0),(np.inf,np.inf,np.inf,2*np.pi)]
        )
    stdev = np.sqrt(np.diag(param_cov))
    fitted = param[0]*t_tr+param[1]+param[2]*np.cos((2*np.pi/T)*t_tr-param[3])
    fitted_df = pd.DataFrame(
        {
            'Time':t,
            'FitTime':t_tr,
            'Fitted':fitted
        }
        ).set_index('Time')
    return (param, stdev, fitted_df)


#if __name__=='__main__':
    #%% test on single point
    #my_point =-500
    
    # select the point qnd compute the trend
    #b = select_point(ds,my_point,verbose=True)
    #df = b['flagged']
    #trend = trend_ols(df.values,df.index.values,summary=False)
    
    
    #%% map the point and plot the the serie
    # extent = [77, 100, 5, 24]
    # map_selected_point(ds,my_point, extent=extent)
    # plot_point(ds,my_point)
    # plot_model(ds,my_point)
    
#%% Plot single point
    point =-300
    plot_point(ds,point)
    #for point in np.arange(-1000, -100, 100):
    extent = [77, 100, 10, 24]
    map_selected_point(ds,point, extent=extent)
    ref=pd.Timestamp(1950,1,1)
    output = select_point(ds, point)
    title = f"Point No: {point} - Distance to Coast: {output['dist']:.2f} km-{output['valid']:.2f}%"
    df = output['ts']
    df_flagged = output['flagged']
        # time = pd.to_datetime(output['ts'].index, unit='D', origin=ref)
    time_flagged = pd.to_datetime(output['flagged'].index, unit='D', origin=ref)
    param, stdev, fitted_model = fit_function(df_flagged, f)
    sla = df_flagged.values.flatten()
    residual = sla - fitted_model['Fitted']

        #Plotting
    fig = plt.figure(figsize=(10, 4))
    plt.plot(time_flagged, sla, '*', color='blue', label="data")
    plt.plot(time_flagged, fitted_model['Fitted'], '--', color='red', label="fit")
    plt.plot(time_flagged, residual, 'o', color='green', label="residual")
    plt.xlabel('year')
    plt.ylabel('Sea level anomaly(m)')
    plt.legend(loc='lower right')
    plt.title(f'slope {param[0]*1000.*365:.1f} mm/yr +/- {stdev[0]*1000.*365:.1f}\namplitude {param[2]*100:.2f} cm +/- {stdev[2]*100:.2f}; phase {np.degrees(param[3]):.1f} degrees +/- {np.degrees(stdev[3]):.1f}')
    #%%
    # plt.show()
        # plt.savefig(f'timeseries_{point}.png')
        
        # # Plot histogram
        # fig, axes = plt.subplots(ncols=2, figsize=(8, 4), sharex=True)
        # axes[0].hist(sla, bins='fd')
        # axes[0].set_title(f'SLA, std={np.std(sla, ddof=1):0.3f}')
        # axes[1].hist(residual, bins='fd')
        # axes[1].set_title(f'Residual, std={np.std(residual, ddof=1):0.3f}')
        # # plt.show()
        # plt.savefig(f'historigram_{point}.png')
        
        #%% Comparison of distribution
        # plt.figure()
        # plt.plot(sla, residual, '*')
        # plt.title(f'Residual vs Origianl SLA @ point {point}')
        # plt.xlabel('SLA')
        # plt.ylabel('Residual')
        # plt.savefig(f'comparison_{point}.png')
    
    #plt.close('all')
    '''monthly mean calculation for ALES'''
    my_point =-400

    b = select_point(ds,my_point,verbose=True)
    dfm = b['flagged']
    ref=pd.Timestamp(1950,1,1)
    t=pd.to_datetime(b['flagged'].index, unit='D', origin=ref)
    dfm['e_year']=t
    dfm['YEAR'] = dfm['e_year'].dt.year
    dfm['month'] = dfm['e_year'].dt.month
    x=dfm.groupby(dfm['e_year'].dt.month).mean()
    mdf=dfm.groupby(dfm['YEAR']).mean()
    mdf=dfm.groupby(dfm['month']).mean()
    dfm.resample('M').mean().plot()