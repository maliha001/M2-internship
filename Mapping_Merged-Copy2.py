#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy import feature
import numpy as np
import pandas as pd
#import nctoolkit as nc
dataset=Path(r'D:\Datasets')


# In[2]:


matplotlib.rcParams['contour.negative_linestyle'] = 'solid'


# In[3]:


bathy_file=dataset / 'etopo5.nc'
msl_file= dataset / 'global_omi_sl_regional_trends_19930101_P20200819.nc'
tg_file=dataset / 'TG_PSML.txt'
#ds=bathy.coarsen({'X':12,'Y':12}).mean()


# In[4]:


bathy=xr.open_dataset(bathy_file)
trend=xr.open_dataset(msl_file)
tg=pd.read_csv(tg_file,sep=';',comment='#',names=['name','lon','lat'])


# In[7]:


trend


# In[5]:


trend['msl_trend'].values


# In[56]:


trend['msl_trend'].plot()
plt.show()


# In[ ]:


##trend.crop(lon = [-(96+31/60), -(89+5/6)], lat = [40 + 36/60, 43 + 30/60])


# In[12]:


#lat_bnds, lon_bnds = [10, 24], [80, 100]
lat=trend.latitude
lat=lat[lat>10]
lat=lat[lat<24]
lon=trend.longitude
lon=lon[lon>80]
lon=lon[lon<100]

trend=trend.sel(latitude=lat,longitude=lon)


# In[9]:




# In[15]:


file1=Path(r"D:\Datasets\74020\ESACCI-SEALEVEL-IND-MSLTR-MERGED-N_INDIAN_JA_014_01-20200603-fv01.1.nc")
file2=Path(r"D:\Datasets\74020\ESACCI-SEALEVEL-IND-MSLTR-MERGED-N_INDIAN_JA_192_01-20200603-fv01.1.nc")
file3=Path(r"D:\Datasets\74020\ESACCI-SEALEVEL-IND-MSLTR-MERGED-N_INDIAN_JA_231_01-20200603-fv01.1.nc")


# In[16]:


ds1=xr.open_dataset(file1)
ds2=xr.open_dataset(file2)
ds3=xr.open_dataset(file3)


# In[17]:


file4=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-014-fv01.1.nc")
file5=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-027-fv01.1.nc")
file6=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-053-fv01.1.nc")
file7=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-090-fv01.1.nc")
file8=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-103-fv01.1.nc")
file9=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-129-fv01.1.nc")
file10=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-166-fv01.1.nc")
file11=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-192-fv01.1.nc")
file12=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-205-fv01.1.nc")
file13=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-231-fv01.1.nc")
file14=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-040-fv01.1.nc")
file15=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-116-fv01.1.nc")
file16=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-079-fv01.1.nc")
file17=Path(r"D:\Datasets\XTRACK\ESACCI-SEALEVEL-L3-SLA-N_INDIAN-MERGED-20200113-JA-155-fv01.1.nc")


# In[18]:


ds4=xr.open_dataset(file4,decode_times=False)
ds5=xr.open_dataset(file5,decode_times=False)
ds6=xr.open_dataset(file6,decode_times=False)
ds7=xr.open_dataset(file7,decode_times=False)
ds8=xr.open_dataset(file8,decode_times=False)
ds9=xr.open_dataset(file9,decode_times=False)
ds10=xr.open_dataset(file10,decode_times=False)
ds11=xr.open_dataset(file11,decode_times=False)
ds12=xr.open_dataset(file12,decode_times=False)
ds13=xr.open_dataset(file13,decode_times=False)
ds14=xr.open_dataset(file14,decode_times=False)
ds15=xr.open_dataset(file15,decode_times=False)
ds16=xr.open_dataset(file16,decode_times=False)
ds17=xr.open_dataset(file17,decode_times=False)


# In[25]:





#definition of color map and levels
cmap_trend = plt.get_cmap('YlOrBr')
cmap = plt.get_cmap('Blues')
reversed_cmap = cmap.reversed()
bathy_levels =[-180,-150,-100,-80,-50,-10,-5]




# basic setup of the map
fig=plt.figure(figsize=(20,20),dpi=80)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([80, 100, 10, 24], ccrs.PlateCarree())



# map the trend and the bathy contour
cs = trend['msl_trend'].plot.contourf(ax=ax,transform=ccrs.PlateCarree(),cmap=cmap_trend,vmin=2,vmax=5,zorder=1,orientation='horizontal')
#cs = bathy['elev'].plot.contourf(ax=ax,transform=ccrs.PlateCarree(),levels=bathy_levels,cmap=reversed_cmap)
#csb = bathy['elev'].plot.contour(ax=ax,transform=ccrs.PlateCarree(),levels=bathy_levels,color=reversed_cmap)

# section to plot the TG data
ax.scatter(tg.lon,tg.lat,transform=ccrs.PlateCarree(),color='green',s=60,zorder=5)

#seanoe
#ax.scatter(ds1.lon.values,ds1.lat.values,transform=ccrs.PlateCarree(),color='cyan',zorder=3) 
#ax.scatter(ds2.lon.values,ds2.lat.values,transform=ccrs.PlateCarree(),color='cyan',zorder=3)  
#ax.scatter(ds3.lon.values,ds3.lat.values,transform=ccrs.PlateCarree(),color='cyan',zorder=3)

#XTRACK
ax.scatter(ds4.lon.values,ds4.lat.values,transform=ccrs.PlateCarree(),color='black',zorder=2) 
ax.scatter(ds5.lon.values,ds5.lat.values,transform=ccrs.PlateCarree(),color='black',zorder=2)
ax.scatter(ds6.lon.values,ds6.lat.values,transform=ccrs.PlateCarree(),color='black',zorder=2)
ax.scatter(ds7.lon.values,ds7.lat.values,transform=ccrs.PlateCarree(),color='black',zorder=2)
ax.scatter(ds8.lon.values,ds8.lat.values,transform=ccrs.PlateCarree(),color='black',zorder=2)
ax.scatter(ds9.lon.values,ds9.lat.values,transform=ccrs.PlateCarree(),color='black',zorder=2)
ax.scatter(ds10.lon.values,ds10.lat.values,transform=ccrs.PlateCarree(),color='black',zorder=2)
ax.scatter(ds11.lon.values,ds11.lat.values,transform=ccrs.PlateCarree(),color='black',zorder=2)
ax.scatter(ds12.lon.values,ds12.lat.values,transform=ccrs.PlateCarree(),color='black',zorder=2)
ax.scatter(ds13.lon.values,ds13.lat.values,transform=ccrs.PlateCarree(),color='black',zorder=2)
ax.scatter(ds14.lon.values,ds14.lat.values,transform=ccrs.PlateCarree(),color='black',)
ax.scatter(ds15.lon.values,ds15.lat.values,transform=ccrs.PlateCarree(),color='black',)
ax.scatter(ds16.lon.values,ds16.lat.values,transform=ccrs.PlateCarree(),color='black',)
ax.scatter(ds17.lon.values,ds17.lat.values,transform=ccrs.PlateCarree(),color='black',)

# section to plot, coastlines, land, gridlines
ax.coastlines(resolution='10m',zorder=4) 


ax.add_feature(cfeature.BORDERS,zorder=4)
ax.add_feature(cfeature.RIVERS,facecolor='blue',zorder=4)
ax.gridlines(draw_labels=True,zorder=7)
ax.add_feature(cfeature.LAND,facecolor="grey",zorder=3)

#cb=fig.colorbar(cs,ax=ax)
#cb.remove()
#plt.colorbar(cs,orientation="horizontal", pad=0.15,fraction=0.15,cbarlabel='Regional Mean Sea Level Trend (mm/yr)')

#divider = make_axes_locatable(ax)
#colorbar_axes = divider.append_axes("right",size="10%",pad=0.1)
#plt.colorbar(cs, cax=colorbar_axes)

#plt.title("Bathymetry over Bay of Bengal")
plt.legend()
plt.plot()



ax.text(82.5,22,"INDIA",style='italic',fontsize=15,fontweight='bold',zorder=6)
ax.text(89,23,"BANGLADESH",style='italic',fontsize=15,fontweight='bold',zorder=6)
ax.text(95,22,"MYANMAR",style='italic',fontsize=15,fontweight='bold',zorder=6)


plt.show()
plt.plot()


# In[89]:


plt.savefig('map.jpg', dpi=300, bbox_inches='tight')

