# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:22:14 2021

@author: BenPurinton
"""
import os
import numpy as np
import itertools
from skimage.util.shape import view_as_windows
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
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

# Size of tiles to calculate metrics on
# NOTE: for tiles larger than ~400x400 pixels the RMSE plane fitting calculation may
# take a long time. For larger tiles (e.g. 20-km or more for 30-m DEMs), switch OFF this metric
tile_size_km = 10

# ON/OFF switch for calculating and plotting only HPHS
only_HPHS = False # True to only do HPHS calc. / False to do dR, RMSE, and HPHS (may be slow!)

# Number of tiles to run metric calculation and plot on
# These are randomly selected from a shuffled list of tiles
# so the metrics are not calculated over the entire DEM
number_of_tiles = 5


#### dR parameters ####

# sigma of gaussian filter
sigma = 0.5 

    
#### RMSE parameters ####

# fitting window: 3=3x3 window, 5=5x5, 7=7x7, etc.
win = 3


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

# %% RUN

# number of cores for multithreading RMSE calcualtion
num_cores = multiprocessing.cpu_count()

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

# loop over tiles, calculating metrics and outputting a plot
for xx, yy in xys:
    
    if not only_HPHS:
        
        # load el
        el = loadDEMclip(dem, xx, yy, tile_size_px)
        
        
        # get the dR metric
        dR = np.abs(el-ndi.gaussian_filter(el, sigma))
        
        
        # get the RMSE (this is slow even with threading, but faster with more threads)
        elb = view_as_windows(el, window_shape=(win, win))
        elb = elb.reshape(elb.shape[0]*elb.shape[1], -1)
        xs = np.array(list(range(win))*win)
        ys = np.array(list(itertools.chain(*[[i]*win for i in range(win)])))
        
        foo = Parallel(n_jobs=num_cores, verbose=1)(delayed(plane_fit_RMSE)(np.array([xs, ys, e]).T) for e in elb)
        rmse = np.empty(el.shape)*np.nan
        new_shape = rmse[win//2:el.shape[0] - win//2, win//2:el.shape[1] - win//2].shape
        rmse[win//2:el.shape[0] - win//2, win//2:el.shape[1] - win//2] = np.reshape(foo, new_shape)
        # NOTE: the final result has NaN values on the edges
            
        
        # get the HPHS metric
        # NOTE in this case we use the step size in meters for calculating the hillshades
        
        # this uses the average step size and would be appropriate for UTM resampled input
        # hphs, hs = HPHS(el, step_meters, kernel, azimuths, angles)
        
        # this is a more correct way using the exact latitude and longitude step size (which differ)
        hphs, hs = HPHS_diff_spacing(el, psx, psy, kernel, azimuths, angles)
    
    
        # plot metrics and hillshade and save out
        fig = plt.figure(figsize=(15, 4), constrained_layout=True)
        gs = fig.add_gridspec(1, 4)
        
        ax = fig.add_subplot(gs[0:1, 0:1])
        im = ax.imshow(hs, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('{}\n{}-km tile ({}, {})'.format(dem_name, tile_size_km, xx, yy), fontsize=10, weight='bold', loc='left', pad=3.5)
        
        ax = fig.add_subplot(gs[0:1, 1:2])
        im = ax.imshow(dR, cmap='viridis', vmin=0, vmax=np.percentile(dR, 99))
        cbar = fig.colorbar(im, ax=ax, shrink=0.5, orientation='vertical')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label(label=r"$dR$ (m)", fontsize=10, labelpad=2)
        cbar.ax.tick_params(labelsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('dR'.format(), fontsize=10, weight='bold', loc='left', pad=3.5)
        
        ax = fig.add_subplot(gs[0:1, 2:3])
        im = ax.imshow(rmse, cmap='plasma', vmin=0, vmax=np.nanpercentile(rmse, 99))
        cbar = fig.colorbar(im, ax=ax, shrink=0.5, orientation='vertical')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label(label=r"$RMSE$ (m)", fontsize=10, labelpad=2)
        cbar.ax.tick_params(labelsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('RMSE'.format(), fontsize=10, weight='bold', loc='left', pad=3.5)
        
        ax = fig.add_subplot(gs[0:1, 3:4])
        im = ax.imshow(hphs, cmap='cividis', vmin=0, vmax=np.nanpercentile(hphs, 99))
        cbar = fig.colorbar(im, ax=ax, shrink=0.5, orientation='vertical')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label(label=r"$HPHS$ (-)", fontsize=10, labelpad=2)
        cbar.ax.tick_params(labelsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('HPHS'.format(), fontsize=10, weight='bold', loc='left', pad=3.5)
    
        file_name = '{}Three_Error_Metrics_{}_{}kmTile_{}_{}.png'.format(figs_out, dem_name, tile_size_km, xx, yy)
        fig.savefig(file_name, dpi=150)
        plt.close()
        
    else:
        # load el
        el = loadDEMclip(dem, xx, yy, tile_size_px)
        
        # get the HPHS metric
        # NOTE in this case we use the step size in meters for calculating the hillshades
        
        # this uses the average step size
        # hphs, hs = HPHS(el, step_meters, kernel, azimuths, angles)
        
        # this is a more correct way using the exact latitude and longitude step size (which differ)
        hphs, hs = HPHS_diff_spacing(el, psx, psy, kernel, azimuths, angles)
    
    
        # plot metrics and hillshade and save out
        fig = plt.figure(figsize=(8, 4), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        
        ax = fig.add_subplot(gs[0:1, 0:1])
        im = ax.imshow(hs, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('{}\n{}-km tile ({}, {})'.format(dem_name, tile_size_km, xx, yy), fontsize=10, weight='bold', loc='left', pad=3.5)
        
        ax = fig.add_subplot(gs[0:1, 1:2])
        im = ax.imshow(hphs, cmap='cividis', vmin=0, vmax=np.nanpercentile(hphs, 99))
        cbar = fig.colorbar(im, ax=ax, shrink=0.5, orientation='vertical')
        cbar.ax.xaxis.set_label_position('top')
        cbar.set_label(label=r"$HPHS$ (-)", fontsize=10, labelpad=2)
        cbar.ax.tick_params(labelsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('HPHS'.format(), fontsize=10, weight='bold', loc='left', pad=3.5)
    
        file_name = '{}HPHS_Error_Metric_{}_{}kmTile_{}_{}.png'.format(figs_out, dem_name, tile_size_km, xx, yy)
        fig.savefig(file_name, dpi=150)
        plt.close()