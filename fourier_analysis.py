# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:52:49 2021

@author: BenPurinton
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from DemErrorFunctions import *


# %% VARIABLES

#### General parameters ####

# DEM to use
dem = 'data/nasadem_s25w068.tif'

# Name for the plots
dem_name = 'SRTM-NASADEM'

# Output folder for plots
figs_out = 'figs/'

# Pixel size in arcsec native resolution
step = 1 # 1 arcsec ~ 30 m pixels

# Size of tiles
tile_size_km = 50

# Number of tiles to run metric calculation and plot on
# These are randomly selected from a shuffled list of tiles
# so the metrics are not calculated over the entire DEM
number_of_tiles = 5


#### HPHS parameters ####

# Solar azimuths and elevation angles (in degrees) for hillshading
azimuths=[0, 90, 180, 270]
angles = [25]

# High-pass kernel
# simple high pass Laplacian with edge sensitivity: 
    # https://stackoverflow.com/questions/32768407/what-is-the-laplacian-mask-kernel-used-in-the-scipy-ndimage-filter-laplace
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])


#### Plotting parameters

# number of bins for plotting/fitting power-law background spectrum
binPL = 20

# number of bins for the maximum envelope
binN = 100

# points to skip when plotting all power-frequency values
# decreasing this number may significantly slow down the figure plotting and saving
skip_val_plot = 3 # only plot every Nth point (e.g., every 3rd point)


# %% RUN

# make the output directory if it doesn't exist
os.makedirs(figs_out, exist_ok=True)

# get a list of shuffled tiles to calculate metrics on
xys, psx, psy, step_meters, tile_size_px = getDEMtiles(dem, tile_size_km)

# NOTE: step_meters is an average of the Latitude/Longitude spacing calculated
# for the given DEMs central Lat/Long, so it may not be exactly e.g., 30 m for a
# 1 arcsec tile.

# The true latitude and longitude spacing using the HPHS calculation is 
# given by psx (longitude spacing) and psy (latitude spacing)


# only take the first n tiles, where n is the number_of_tiles variable
xys = xys[0:number_of_tiles]

for xx, yy in xys:
    
    # load el
    el = loadDEMclip(dem, xx, yy, tile_size_px)
    
    # get the HPHS metric
    # NOTE in this case we use the step size in meters for calculating the hillshades
    
    # this uses the average step size and would be appropriate for UTM resampled input
    # hphs, hs = HPHS(el, step_meters, kernel, azimuths, angles)
    
    # this is a more correct way using the exact latitude and longitude step size (which differ)
    hphs, hs = HPHS_diff_spacing(el, psx, psy, kernel, azimuths, angles)
    
    # get 2D FFT
    # NOTE this is calculated using the step size in arcsec, not with uneven
    f1d, f2d, p1d, p2d, F_ang1d, F_ang2d = doDFT(hphs, step)
    
    # norm
    bin_center, bin_med, pl_fit, fit, p1d_norm, p2d_norm = fftNorm(f1d, f2d, p1d, p2d, bins=binPL)
    
    # make maximum binned envelope (just using 100 bins as default, but could be changed above)
    # this binning gets the exact bin edges and bin centers with log-spaced bins
    # NOTE the bins are only calculated down to the Nyquist frequency (twice the pixel size)
    wvl_bins = np.logspace(np.log10(step*2), np.log10(1/f1d.min()), binN * 2 - 1)
    wvl_bins_c = wvl_bins[1::2]
    wvl_bins = wvl_bins[::2]
    
    # using pandas to do the max binning
    tmp_df = pd.DataFrame({'wv1d' : 1/f1d, 'p1d_norm' : p1d_norm})
    tmp_df['wvl_bins'] = pd.cut(tmp_df['wv1d'], wvl_bins)
    peak_env = tmp_df.groupby(tmp_df['wvl_bins'])['p1d_norm'].max().values
    

    # plot it all!
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8),
                                   gridspec_kw={'left':0.1, 'right':0.95, 
                                                'bottom':0.1, 'top':0.95})
    ax1, ax2, ax3, ax4 = axes[0, 1], axes[1, 1], axes[0, 0], axes[1, 0]
    
    # first plot the original 1D power specturm
    ax1.loglog(1/f1d[::skip_val_plot][::-1], p1d[::skip_val_plot][::-1], 
               '.', c='gray', alpha=0.5, markersize=3, label="Power", rasterized=True)
    ax1.loglog(1/bin_center[::-1], bin_med[::-1], 'ko--', markersize=8, alpha=1, 
              label="Log bins at median")
    ax1.loglog(1/bin_center[::-1], pl_fit[::-1], 'r-', lw=1, label="Power-law fit to bins")
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_xticklabels(["{:,.0f}\n[{:g}]".format(a, 1/a) for a in ax1.get_xticks()])
    ax1.set_xlim(np.max(1/f1d), step)
    ax1.set_xlabel('Wavelength (arcsec) / [Frequency (arcsec$^{-1}$)]', fontsize=12)
    ax1.set_ylabel('Mean Squared Amplitude', fontsize=12)
    ax1.grid(True, which="both", lw=0.3)
    ax1.legend(loc='lower left', fontsize=10)

    # then plot the normalized 1D power specturm
    ax2.semilogx(1/f1d[::skip_val_plot][::-1], p1d_norm[::skip_val_plot][::-1], '.', 
                 c='gray', alpha=0.5, markersize=8, rasterized=True, label="__nolabel__")
    ax2.semilogx(wvl_bins_c, peak_env, '-', lw=1, label='Max. envelope ({} bins)'.format(len(wvl_bins_c)+1))
    ax2.set_ylim(0, p1d_norm.max()+1)
    ax2.set_xticks(ax2.get_xticks())
    ax2.set_xticklabels(["{:,.0f}\n[{:g}]".format(a, 1/a) for a in ax2.get_xticks()])
    ax2.set_xlim(np.max(1/f1d), step)
    ax2.set_xlabel('Wavelength (arcsec) / [Frequency (arcsec$^{-1}$)]', fontsize=12)
    ax2.set_ylabel('Normalized Amplitude', fontsize=12)
    ax2.grid(True, which="both", lw=0.3)
    ax2.legend(loc='upper left', fontsize=10)

    # now plot the original HPHS grid
    im = ax3.imshow(hphs, cmap='cividis', interpolation='nearest', vmax=np.percentile(hphs, 99.9))
    cbar = plt.colorbar(im, ax=ax3, shrink=0.5)
    cbar.set_label('HPHS', fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title('{}; {}-km TILE {}, {}'.format(dem_name, tile_size_km, xx, yy), fontsize=12)
    
    # also plot the normalized 2D power spectrum
    im = ax4.imshow(p2d_norm, interpolation='nearest', cmap='magma_r', vmax=np.nanpercentile(p2d_norm, 99.9))
    fig.colorbar(im, ax=ax4, label="Normalized Power (color to 99.9th perc.)", shrink=0.5)
    ax4.set_xlabel("X Frequency (arcsec$^{-1}$)", fontsize=12)
    ax4.set_ylabel("Y Frequency (arcsec$^{-1}$)", fontsize=12)
    nfy, nfx = f2d.shape
    nyq = f2d[nfy//2 + 1, 0]
    n_labels = 8
    ax4.set_xticks(np.linspace(1, nfx, n_labels))
    ax4.set_yticks(np.linspace(1, nfy, n_labels))
    ticks = ["{:.3f}".format(a) for a in np.linspace(-nyq, nyq, n_labels)]
    ax4.set_xticklabels(ticks)
    ax4.set_yticklabels(ticks)

    file_name = '{}DFT_{}_{}kmTile_{}_{}.png'.format(figs_out, dem_name, tile_size_km, xx, yy)
    fig.savefig(file_name, dpi=150)
    plt.close()

