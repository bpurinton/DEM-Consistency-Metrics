# DEM-Consistency-Metrics

This is a collection of tools to evaluate inter-pixel consistency metrics on gridded digital elevation models (DEMs) as outlined in:

  > Purinton, B. and Bookhagen, B, Beyond vertical point accuracy: Assessing inter-pixel consistency in 30 m  global DEMs for the arid Central Andes, 2021, Frontiers in Earth Science, _submitted_.


![](img/srtm-nasadem-example-HPHS.png)

# Code Description

All utility functions are found in `DemErrorFunctions.py`.

A script to calculate and plot three error metrics described in the paper is `error_metrics.py`. These metrics are:

* Gaussian smoothing and differencing, _dR_ (m)
* Plane fit residuals, _RMSE_ (m)
* High-pass hillshade filtering, _HPHS_ (-)

A script to tile and run the Fourier frequency analysis on the open access SRTM-NASADEM (https://lpdaac.usgs.gov/products/nasadem_hgtv001) is `fourier_analysis.py`. The DEM is provided in the `data` directory (2 by 2 degree, four merged tiles at native 1 arcsec resolution in WGS84/EGM96 horizontal/vertical datum).
