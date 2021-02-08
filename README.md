# M2-internship

blablabla

#Mapping CMEMS datasets

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

ds=xr.open_dataset(CMEMS)

%matplotlib inline
fig=plt.figure(figsize=(15,20),dpi=150)
cmap = plt.get_cmap('YlOrBr')
levels = [0., 0.5, 1, 1.5, 2., 2.5, 3., 4., 5., 6.]
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent([80, 100, 10, 24], ccrs.PlateCarree())
cs = ds['msl_trend'].plot.contourf(ax=ax,transform=ccrs.PlateCarree(),levels=levels,cmap=cmap)
ax.coastlines(resolution='10m') 
ax.gridlines(draw_labels=True)
#plt.title(ds.msli_trend.long_name)
plt.legend()
plt.show()

