import random
import os
import math
import numpy as np
from osgeo import gdal, osr
import scipy
from scipy import stats
from scipy import ndimage as ndi
from pyproj import Proj, CRS
from pysheds.grid import Grid
import rasterio
from matplotlib import pyplot as plt
import matplotlib.colors as mplcol


def getDEMtiles(dem, tile_size_km):
    """
    Loads a full DEM and produces a list of tile coordinates for generating 
    square tiles of a desired size (in km).

    Parameters
    ----------
    dem : string
        Name, including relative file path to DEM file.
    tile_size_km : int
        Size of tiles to produce in kilometers.

    Returns
    -------
    xys : list
        List of tuples with (column, row) coordinates of upper left pixels.
    psx : float
        Approximate pixel size in longitude in meters.
    psy : float
        Approximate pixel size in latitude in meters.
    step_meters : float
        Approximate pixel size in meters.
    tile_size_px : int
        Size of tiles converted from kilometers to pixels based on pixel size.

    """
    ds = gdal.Open(dem)
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    nan = band.GetNoDataValue()

    # read as array and set NaN
    el = band.ReadAsArray().astype(float)
    el[el == nan] = np.nan

    print('getting {}-km tiles from {}\nwith original shape {}'\
          .format(tile_size_km, os.path.split(dem)[1], el.shape))

    # get pixel size
    cols = el.shape[1]
    rows = el.shape[0]
    minx, maxy = gt[0], gt[3]
    maxx, miny = gt[0] + gt[1] * cols, gt[3] + gt[5] * rows

    # read crs
    crs = CRS.from_wkt(ds.GetProjection()).to_epsg()

    # get step in m if geographic projection
    if crs == 4326:
        epsg_code = convert_wgs_to_utm(minx, miny)
        pp = Proj('EPSG:{}'.format(epsg_code))
        proj = CRS.from_epsg(epsg_code).to_wkt()
        minx, miny = pp(minx, miny)
        maxx, maxy = pp(maxx, maxy)
    psx = (maxx - minx) / cols
    psy = (maxy - miny) / rows
    step_meters = np.round((psx + psy) / 2, 0)

    # close dataset
    ds = None

    # get potential tiles (large then small)
    tile_size_px = int(np.round(tile_size_km * 1000 / step_meters, 0))
    xys = []
    for xx in range(0, cols-tile_size_px, tile_size_px):
        for yy in range(0, rows-tile_size_px, tile_size_px):
            xys.append((xx, yy))
    random.shuffle(xys)

    print('made {} tiles'.format(len(xys)))

    return xys, psx, psy, step_meters, tile_size_px


def loadDEMclip(dem, xx, yy, tile_size_px):
    """
    Takes a DEM file and row and column coordinates and clips a square tile from
    the DEM, returning the clip as a numpy array in memory. Nothing is written to
    disk.

    Parameters
    ----------
    dem : string
        Name, including relative file path to DEM file.
    xx : int
        Column coordinate of upper left pixel.
    yy : int
        Row coordinate of upper left pixel.
    tile_size_px : int
        Size of square tile in pixels.

    Returns
    -------
    el : numpy array
        Array clipped from DEM file.

    """
    kwargs = {'format' : 'VRT',
              'srcWin' : [xx, yy, tile_size_px, tile_size_px]}

    ds = gdal.Translate('', dem, **kwargs)
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    nan = band.GetNoDataValue()
    # print(nan)

    # read as array and set NaN
    el = band.ReadAsArray().astype(float)
    el[el == nan] = np.nan

    # close dataset
    ds = None

    return el


def np_slope(z, d):
    """
    https://github.com/UP-RS-ESP/TopoMetricUncertainty/blob/master/uncertainty.py
    Provides slope in degrees.
    """
    dy, dx = np.gradient(z, d)
    slope = np.arctan(np.sqrt(dx*dx+dy*dy))*180/np.pi
    return slope


def np_aspect(z, d):
    """
    Outputs terrain aspect in degrees with North = 0; East = 90; South = 180; West = 270
    See:
    https://github.com/UP-RS-ESP/TopoMetricUncertainty/blob/master/uncertainty.py
    and
    https://github.com/USDA-ARS-NWRC/topocalc/blob/main/topocalc/gradient.py
    and
    https://github.com/LSDtopotools/LSD_Resolution/blob/a3ff6af7dc3fc865c838ce6eb968866431b80352/LSDRaster.cpp
    """
    dy, dx = np.gradient(z, d)
    a = 180 * np.arctan2(dy, -dx) / np.pi
    aspect = 90 - a
    aspect[a < 0] = 90 - a[a < 0]
    aspect[a > 90] = 360 - a[a > 90] + 90
    idx = (dy==0) & (dx==0)
    aspect[idx] = 180

    return aspect


def hillshade(array, spacing, azimuth=315, angle_altitude=20):
    """
    This function is used to generate a hillshade of the topography. It produces
    identical outputs to 'gdaldem hillshade -alg ZevenbergenThorne' (<--this was tested)
    From here: https://github.com/LSDtopotools/LSDTopoTools_CRNBasinwide/blob/master/LSDRaster.cpp
    """
    slope = np_slope(array, spacing)*np.pi/180
    aspect = np_aspect(array, spacing)*np.pi/180

    # This bit isn't necessary with above np_aspect output (0 North; 90 East)
    # azimuth_math = 360 - azimuth + 90
    # if azimuth_math >= 360.0:
    #     azimuth_math = azimuth_math - 360

    azimuth_math = azimuth
    azimuthrad = azimuth_math * np.pi /180.0

    zenith_rad = (90 - angle_altitude) * np.pi / 180.0

    shaded = (np.cos(zenith_rad) * np.cos(slope)) + (np.sin(zenith_rad) * np.sin(slope) * np.cos((azimuthrad) - aspect))

    shaded = 255*(shaded + 1)/2

    return shaded.astype(int)


def HPHS(el, step, kernel, azimuths, angles):
    """
    Calculate HPHS metric

    Parameters
    ----------
    el : numpy array
        Elevation values in array.
    step : float
        Average pixel spacing in meters.
    kernel : numpy array
        High pass filtering kernel.
    azimuths : list
        Sun azimuths.
    angles : list
        Sun elevation angles.

    Returns
    -------
    hphs : numpy array
        High-pass hillshade metric.
    hs : numpy array
        Hillshade image.

    """

    highpasses = np.zeros((el.shape[0], el.shape[1], len(azimuths), len(angles)))
    for ang_num, ang in enumerate(angles):
        for az_num, az in enumerate(azimuths):
            # HS
            hs = hillshade(el, step, azimuth=az, angle_altitude=ang)
            # edge filter
            hp = abs(ndi.convolve(hs, kernel))
            highpasses[:,:,az_num,ang_num] = hp[:]

    # take maximum value from rotated stack
    hphs = np.nanmax(highpasses, axis=2)[:,:, 0].astype(int)

    return hphs, hs



def np_slope_diff_spacing(z, xspace, yspace):
    """
    https://github.com/UP-RS-ESP/TopoMetricUncertainty/blob/master/uncertainty.py
    Provides slope in degrees.
    """
    dy, dx = np.gradient(z, xspace, yspace)
    return np.arctan(np.sqrt(dx*dx+dy*dy))*180/np.pi


def np_aspect_diff_spacing(z, xspace, yspace):
    """
    Outputs terrain aspect in degrees with North = 0; East = 90; South = 180; West = 270
    See:
    https://github.com/UP-RS-ESP/TopoMetricUncertainty/blob/master/uncertainty.py
    and
    https://github.com/USDA-ARS-NWRC/topocalc/blob/main/topocalc/gradient.py
    and
    https://github.com/LSDtopotools/LSD_Resolution/blob/a3ff6af7dc3fc865c838ce6eb968866431b80352/LSDRaster.cpp
    """
    dy, dx = np.gradient(z, xspace, yspace)
    a = 180 * np.arctan2(dy, -dx) / np.pi
    aspect = 90 - a
    aspect[a < 0] = 90 - a[a < 0]
    aspect[a > 90] = 360 - a[a > 90] + 90
    idx = (dy==0) & (dx==0)
    aspect[idx] = 180

    return aspect

def hillshade_diff_spacing(array, xspace, yspace, azimuth=315, angle_altitude=20):
    """
    This function is used to generate a hillshade of the topography. It produces
    identical outputs to 'gdaldem hillshade -alg ZevenbergenThorne' (<--this was tested)
    From here: https://github.com/LSDtopotools/LSDTopoTools_CRNBasinwide/blob/master/LSDRaster.cpp
    """
    slope = np_slope_diff_spacing(array, xspace, yspace)*np.pi/180
    aspect = np_aspect_diff_spacing(array, xspace, yspace)*np.pi/180

    # This bit isn't necessary with above np_aspect output (0 North; 90 East)
    # azimuth_math = 360 - azimuth + 90
    # if azimuth_math >= 360.0:
    #     azimuth_math = azimuth_math - 360

    azimuth_math = azimuth
    azimuthrad = azimuth_math * np.pi /180.0

    zenith_rad = (90 - angle_altitude) * np.pi / 180.0

    shaded = (np.cos(zenith_rad) * np.cos(slope)) + (np.sin(zenith_rad) * np.sin(slope) * np.cos((azimuthrad) - aspect))

    shaded = 255*(shaded + 1)/2

    return shaded.astype(int)


def HPHS_diff_spacing(el, xspace, yspace, kernel, azimuths, angles):
    """
    Calculate HPHS metric

    Parameters
    ----------
    el : numpy array
        Elevation values in array.
    xspace : float
        Longitudinal pixel spacing in meters.
    yspace : float
        Latitudinal pixel spacing in meters.
    kernel : numpy array
        High pass filtering kernel.
    azimuths : list
        Sun azimuths.
    angles : list
        Sun elevation angles.

    Returns
    -------
    hphs : numpy array
        High-pass hillshade metric.
    hs : numpy array
        Hillshade image.

    """

    highpasses = np.zeros((el.shape[0], el.shape[1], len(azimuths), len(angles)))
    for ang_num, ang in enumerate(angles):
        for az_num, az in enumerate(azimuths):
            # HS
            hs = hillshade_diff_spacing(el, xspace, yspace, azimuth=az, angle_altitude=ang)
            # edge filter
            hp = abs(ndi.convolve(hs, kernel))
            highpasses[:,:,az_num,ang_num] = hp[:]

    # take maximum value from rotated stack
    hphs = np.nanmax(highpasses, axis=2)[:,:, 0].astype(int)

    # This normalization should not be done for inter DEM comparisons
    # hphs = hphs / hphs.max()

    return hphs, hs


def plane_fit_RMSE(points):
    """
    Simple function returns RMSE of fit plane to window. For higher order fitting
    functions see curvFit_lstsq_polynom function below

    Parameters
    ----------
    points : array
        (X, Y, Z) array of coordinates. Each row is an (X, Y, Z) triple.

    Returns
    -------
    p1_rmse : float
        Root mean square error of plane fit.

    """
    ctr = points.mean(axis=0)
    points = points - ctr
    A = np.c_[points[:,0], points[:,1], np.ones(points.shape[0])]
    p1_C,_,_,_ = scipy.linalg.lstsq(A, points[:,2], lapack_driver='gelsy')    # coefficients
    Z_pts = p1_C[0]*points[:,0] + p1_C[1]*points[:,1] + p1_C[2]
    p1_dz = points[:,2] - Z_pts
    mse = (np.square(p1_dz)).mean(axis=0)
    p1_rmse = np.sqrt(mse)

    return p1_rmse


def curvFit_lstsq_polynom(points, order=2):
    """
    Surface fitting to 3D pointcloud. Order=1 assumes a linear plane and uses 
    a least squared approach. Higher order surfaces uses quadratic curve 
    fitting approaches: Fitting a second order polynom to a point cloud and 
    deriving the curvature in a simplified form. We follow: 
    Evans, I. S. (1980), An integrated system of terrain analysis and slope 
    mapping, Z. Geomorphol., 36, 274–295.
    More details: https://gis.stackexchange.com/questions/37066/how-to-calculate-terrain-curvature
    Original functions by B. Bookhagen: https://github.com/UP-RS-ESP/PC_geomorph_roughness/blob/master/pc_geomorph_roughness.py#L97

    Parameters
    ----------
    points : array
        (X, Y, Z) array of coordinates. Each row is an (X, Y, Z) triple.
    order : int, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    Coefficients (_C), residuals (_dz), root mean square errors (_rmse),
    local slope (_slope), local aspect (_aspect), total curvature (_Curvature),
    contour curvature (_curv_contour), tangential curvature (_curv_tan), and
    profile curvature (_curv_profc), for each of the selected polynomial fitting
    orders (p1, p2, and/or p4).
    """
    #commented this out, because it will slow down processing.
    #points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    #try:
    #    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    #except AssertionError:
    #    return np.nan, np.nan, np.nan

    #points = points.T
    #nr_of_points = len(points)
    ctr = points.mean(axis=0)
    points = points - ctr

    # if points.shape[0] > 8:
    if order==1:
        #1st order
        A = np.c_[points[:,0], points[:,1], np.ones(points.shape[0])]
        p1_C,_,_,_ = scipy.linalg.lstsq(A, points[:,2], lapack_driver='gelsy')    # coefficients
        Z_pts = p1_C[0]*points[:,0] + p1_C[1]*points[:,1] + p1_C[2]
        p1_dz = points[:,2] - Z_pts
        mse = (np.square(p1_dz)).mean(axis=0)
        p1_rmse = np.sqrt(mse)
        p1_slope = np.sqrt( p1_C[0]**2 + p1_C[1]**2 )
        #Calculate unit normal vector of fitted plane:
        #N = C / np.sqrt(sum([x*x for x in p1_C]))
        p1_aspect = np.rad2deg(np.arctan2(p1_C[1], -p1_C[0]))
        if p1_aspect < 0:
            p1_aspect = 90.0 - p1_aspect
        elif p1_aspect > 90.0:
            p1_aspect = 360.0 - p1_aspect + 90.0
        else:
            p1_aspect = 90.0 - p1_aspect

        # return p1_C, p1_dz, p1_rmse,p1_slope, p1_aspect

    else:
        p1_C, p1_dz, p1_rmse,p1_slope, p1_aspect = np.nan, np.nan, np.nan, np.nan, np.nan,


    # if points.shape[0] > 8:
    if order==2:
        #2nd order
        # best-fit quadratic curve
        #Z = Dx² + Ey² + Fxy + Gx + Hy + I
        #z = r*x**2 + t * y**2 + s*x*y + p*x + q*y + u
        A = np.c_[points[:,0]**2., \
                  points[:,1]**2., \
                  points[:,0]*points[:,1], \
                  points[:,0], points[:,1], np.ones(points.shape[0])]
        p2_C,_,_,_ = scipy.linalg.lstsq(A, points[:,2], lapack_driver='gelsy')    # coefficients
        Z_pts = p2_C[0]*points[:,0]**2. + p2_C[1]*points[:,1]**2. + p2_C[2]*points[:,0]*points[:,1] + p2_C[3]*points[:,0] + p2_C[4]*points[:,1] + p2_C[5]
        p2_dz = points[:,2] - Z_pts
        mse = (np.square(p2_dz)).mean(axis=0)
        p2_rmse = np.sqrt(mse)
        #dZ_residuals = np.linalg.norm(errors)
        fxx=p2_C[0]
        fyy=p2_C[1]
        fxy=p2_C[2]
        fx=p2_C[3]
        fy=p2_C[4]
        #mean curvature (arithmetic average)
        c_m = - ( (1 + (fy**2))*fxx - 2*fxy*fx*fy+ (1 + (fx**2))*fyy ) / (2*( (fx**2) + (fy**2) + 1)**(3/2) )

        #tangential (normal to gradient) curvature
        c_t = - ( ( fxx*(fy**2) - 2*fxy * fx * fy + fyy * (fx**2) ) / ( ( (fx**2) + (fy**2) ) * ((fx**2) + (fy**2) + 1)**(1/2) ) )

        #difference (range of profile and tangential)
        c_d = c_m - c_t

        #profile (vertical or gradient direction) curvature
        c_p = c_m + c_d

        #contour (horizontal or contour direction)
        c_c = - ( ( fxx * (fx**2) - 2 * fxy * fx * fy + fyy * (fx**2) ) / ( ( (fx**2) + (fy**2) )**(3/2) ) )

        #Curvature = 2*fxx + 2*fyy
        p2_Curvature = c_m
        #curv_contour = Curvature
        p2_curv_contour = c_c
        p2_curv_tan = c_t
        p2_curv_profc = c_p
        p2_slope = np.sqrt( fx**2 + fy**2 )
        #N = p2_C[3::] / np.sqrt(sum([x*x for x in p2_C[3::]]))
        #azimuth = np.degrees(np.arctan2(N[1], N[0])) + 180
        p2_aspect = np.rad2deg(np.arctan2(fy, -fx))
        if p2_aspect < 0:
            p2_aspect = 90.0 - p2_aspect
        elif p2_aspect > 90.0:
            p2_aspect = 360.0 - p2_aspect + 90.0
        else:
            p2_aspect = 90.0 - p2_aspect

        # return p2_C, p2_dz, p2_rmse, p2_slope, p2_aspect, p2_Curvature, p2_curv_contour, p2_curv_tan, p2_curv_profc

    else:
        p2_C, p2_dz, p2_rmse, p2_slope, p2_aspect, p2_Curvature, p2_curv_contour, p2_curv_tan, p2_curv_profc = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # if points.shape[0] > 8:
    if order==4:
        #4th order
        # best-fit fourth-order polynomial
        #Z = Ax²y² + Bx²y + Cxy² + Dx² + Ey² + Fxy + Gx + Hy + I
        #A = [(Z1 + Z3 + Z7 + Z9) / 4 - (Z2 + Z4 + Z6 + Z8) / 2 + Z5] / L4
        #B = [(Z1 + Z3 - Z7 - Z9) /4 - (Z2 - Z8) /2] / L3
        #C = [(-Z1 + Z3 - Z7 + Z9) /4 + (Z4 - Z6)] /2] / L3
        #D = [(Z4 + Z6) /2 - Z5] / L2
        #E = [(Z2 + Z8) /2 - Z5] / L2
        #F = (-Z1 + Z3 + Z7 - Z9) / 4L2
        #G = (-Z4 + Z6) / 2L
        #H = (Z2 - Z8) / 2L
        #I = Z5
        A = np.c_[points[:,0]**2. * points[:,1]**2., \
                  points[:,0]**2. * points[:,1], \
                  points[:,0] * points[:,1]**2., \
                  points[:,0]**2., \
                  points[:,1]**2., \
                  points[:,0]*points[:,1], \
                  points[:,0], points[:,1], \
                  np.ones(points.shape[0]) ]
        p4_C,_,_,_ = scipy.linalg.lstsq(A, points[:,2], lapack_driver='gelsy')    # coefficients
        Z_pts = p4_C[0]*(points[:,0]**2.) * (points[:,1]**2.) \
            + p4_C[1]*(points[:,0]**2.) * points[:,1] \
            + p4_C[2]*points[:,0] * (points[:,1]**2.) \
            + p4_C[3]*(points[:,0]**2.) + p4_C[4]*points[:,1]**2. \
            + p4_C[5]*points[:,0] * points[:,1] \
            + p4_C[6]*points[:,0] + p4_C[7]*points[:,1] + p4_C[8]
        p4_dz = points[:,2] - Z_pts
        mse = (np.square(p4_dz)).mean(axis=0)
        p4_rmse = np.sqrt(mse)
        #dZ_residuals = np.linalg.norm(errors)
        fx=p4_C[6]
        fy=p4_C[7]
        fxx=p4_C[3]
        fxy=p4_C[5]
        fyy=p4_C[4]
        #mean curvature (arithmetic average)
        c_m = - ( (1 + (fy**2))*fxx - 2*fxy*fx*fy+ (1 + (fx**2))*fyy ) / (2*( (fx**2) + (fy**2) + 1)**(3/2) )

        #tangential (normal to gradient) curvature
        c_t = - ( ( fxx*(fy**2) - 2*fxy * fx * fy + fyy * (fx**2) ) / ( ( (fx**2) + (fy**2) ) * ((fx**2) + (fy**2) + 1)**(1/2) ) )

        #difference (range of profile and tangential)
        c_d = c_m - c_t

        #profile (vertical or gradient direction) curvature
        c_p = c_m + c_d

        #contour (horizontal or contour direction)
        c_c = - ( ( fxx * (fx**2) - 2 * fxy * fx * fy + fyy * (fx**2) ) / ( np.sqrt( ( (fx**2) + (fy**2) )**(2) ) ) )

    #        p = fx
    #        q = fy
    #        r = fxx
    #        s = fxy
    #        t = fyy
    #        curv_k_h = - ( ( (q**2) * r - 2*p*q*s + (p**2) * t) / ( ((p**2) + (q**2)) * np.sqrt(1 + (p**2) + (q**2)) ) )
    #        curv_k_v = - ( ( (p**2) * r + 2*p*q*s + (q**2) * t) / ( ((p**2) + (q**2)) * np.sqrt( (1 + (p**2) + (q**2))**3 ) ) )

        #Curvature = 2*fxx + 2*fyy
        p4_Curvature = c_m
        #curv_contour = Curvature
        p4_curv_contour = c_c
        p4_curv_tan = c_t
        p4_curv_profc = c_p
        p4_slope = np.sqrt( fx**2 + fy**2 )
        #N = p4_C[6::] / np.sqrt(sum([x*x for x in p4_C[6::]]))
        p4_aspect = np.rad2deg(np.arctan2(fy, -fx))
        if p4_aspect < 0:
            p4_aspect = 90.0 - p4_aspect
        elif p4_aspect > 90.0:
            p4_aspect = 360.0 - p4_aspect + 90.0
        else:
            p4_aspect = 90.0 - p4_aspect

        # return p4_C, p4_dz, p4_rmse, p4_slope, p4_aspect, p4_Curvature, p4_curv_contour, p4_curv_tan, p4_curv_profc

    else:
        p4_C, p4_dz, p4_rmse, p4_slope, p4_aspect, p4_Curvature, p4_curv_contour, p4_curv_tan, p4_curv_profc = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return p1_C, p1_dz, p1_rmse,p1_slope, p1_aspect, \
            p2_C, p2_dz, p2_rmse, p2_slope, p2_aspect, p2_Curvature, p2_curv_contour, p2_curv_tan, p2_curv_profc, \
            p4_C, p4_dz, p4_rmse, p4_slope, p4_aspect, p4_Curvature, p4_curv_contour, p4_curv_tan, p4_curv_profc 

# %%

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mplcol.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def CalculateChannelSlope(pts_array, slope_window_size=5):
    """
    Clubb 2019 - JGR:ES
    Modified from: https://github.com/UP-RS-ESP/river-clusters/blob/master/clustering.py
    """

    grad = np.empty(len(pts_array))
    slicer = (slope_window_size - 1)/2

    for index, x in enumerate(pts_array):
        start_idx = index-slicer
        if start_idx < 0:
            start_idx=0
        end_idx = index+slicer+1
        if end_idx > len(pts_array):
            end_idx = len(pts_array)
        # find the rows above and below relating to the window size. We use whatever nodes
        # are available to not waste the data.
        this_slice = pts_array[int(start_idx):int(end_idx)]
        # now regress this slice
        x = this_slice[:,0]
        y = this_slice[:,1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        grad[index] = abs(slope)

    return grad

# %%



# %%
def doDFT(arr, step):
    # https://github.com/bpurinton/DEM-FFT/blob/master/example_analysis.ipynb
    ny, nx = arr.shape

    # fit plane and remove trend
    x, y = np.meshgrid(range(nx), range(ny))
    A = np.vstack([x.ravel(), y.ravel(), np.ones(len(x.ravel()))]).T
    fit = np.linalg.lstsq(A, arr.ravel(), rcond=None)[0]
    arr_fft = arr - (fit[0]*x + fit[1]*y + fit[2])

    # apply hanning windowing to reduce spectral leakage on edges
    hann_y = np.hanning(ny)
    hann_x = np.hanning(nx)
    hann_2d = np.sqrt(np.outer(hann_y, hann_x))
    hann_weight = np.sum(hann_2d ** 2)
    arr_fft = arr_fft * hann_2d

    # This next step is done to optimize the Cooley and Turkey (1965)
    # Discrete Fourier Transfom (DFT) method used by numpy, which operates
    # most efficiently when the length scales are powers of 2 and the grid
    # is square
    Lx = int(2**(np.ceil(np.log(np.max((nx, ny)))/np.log(2))))
    Ly = Lx
    # Lx, Ly = nx, ny

    # frequency increments
    # dfx = 1/(step*Lx)
    # dfy = 1/(step*Ly)

    # run the fft
    fft = np.fft.fftn(arr_fft, (Ly, Lx))

    # shift zero frequency to center
    fft_shift = np.fft.fftshift(fft)

    # # index of zero frequency (DC component)
    xc, yc = (Lx//2, Ly//2)

    # # zero out the DC component
    fft_shift[yc, xc] = 0

    # get the DFT periodogram with units of m^2 for topography
    # include weights of hann to correct for windowing
    p2d = np.abs(fft_shift)**2 / (Lx * Ly * hann_weight)

    # The periodogram is a measure of how much of the
    # original elevation field's variance falls within a given frequency range.
    # You can check that the sum of the periodogram is roughly equal to the
    # variance in Z. (The variance will be somewhat less due to the zero padding.)

    # calculate radial frequencies
    # xc, yc = (Lx//2, Ly//2) # (Lx//2 + 1, Ly//2 - 1) # center coordinate
    x, y = np.meshgrid(range(Lx), range(Ly))#[::-1])

    # wavenumbers
    kx = x - xc
    ky = y - yc
    # kx_, ky_ = np.meshgrid(range(-Lx//2, Lx//2 - 1), range(Ly//2, -Ly//2+1, -1))

    # radial frequencies
    fx = kx / (Lx * step)
    fy = ky / (Ly * step)
    f2d = np.sqrt(fx**2 + fy**2)
    # w2d = np.sqrt((1/fx)**2 + (1/fy)**2)
    # f2d = 1/w2d

    # fourier angles
    F_ang2d = np.rad2deg(np.arctan2(ky*step, kx*step))

    # Create sorted, non-redundant vectors of frequency and power
    p1d = p2d[:, 0:xc+1].copy() # only half the power (reflected across the center)
    f1d = f2d[:, 0:xc+1].copy() # same for the frequency
    F_ang1d = F_ang2d[:, 0:xc+1].copy() # same for angle

    # set reundant columns to negative for clipping below
    f1d[yc:Ly, xc] = -1

    # concatenate frequency and power and sort by frequency
    f1d = np.c_[f1d.ravel(), p1d.ravel(), F_ang1d.ravel()]
    I = np.argsort(f1d[:, 0])
    f1d = f1d[I, :]

    # remove negative values
    f1d = f1d[f1d[:, 0] > 0, :]

    # extract power, angle, and frequency (factor of 2 corrects for taking half the spectrum)
    p1d = 2 * f1d[:, 1] # the sum of the p2d and p1d should now be approximately equal
    F_ang1d = f1d[:, 2]
    f1d = f1d[:, 0]

    return f1d, f2d, p1d, p2d, F_ang1d, F_ang2d


def fftNorm(f1d, f2d, p1d, p2d, bins=20):

    # bin the data using log bins
    bins = 20
    f_bins = np.logspace(np.log10(f1d.min()), np.log10(f1d.max()), bins * 2 - 1)
    bin_med, edges, _ = stats.binned_statistic(f1d, p1d, statistic=np.nanmedian,
                                               bins=f_bins[::2])
    # bin_center = edges[:-1] + np.diff(edges)/2
    bin_center = f_bins[1::2]

    # sometimes NaN values remain in some bins, throw those bins out
    bin_center = bin_center[np.isfinite(bin_med)]
    bin_med = bin_med[np.isfinite(bin_med)]

    # apply a power-law fit to the bins
    A = np.vstack([np.log10(bin_center), np.ones(len(bin_center))]).T
    fit = np.linalg.lstsq(A, np.log10(bin_med), rcond=None)[0]
    pl_fit = (10**fit[1]) * (bin_center**fit[0])

    with np.errstate(divide='ignore', invalid='ignore'):
        # use the power-law fit to normalize the 1D spectrum
        p1d_norm = p1d / ((10**fit[1]) * (f1d**fit[0]))

        # use the power-law fit to normalize the 2D spectrum
        p2d_norm = p2d / ((10**fit[1]) * (f2d**fit[0]))

    return bin_center, bin_med, pl_fit, fit, p1d_norm, p2d_norm

def plotDFT_blob(file_name, dem_name, slp_asp_label, label, f1d, p1d, bin_center, bin_med, pl_fit,
            maxWVL, step, p1d_norm, p2d_norm, hphs_, maxP,
            f2d, F_ang1d_maxP, wvl_maxP, s_wvl, s_power, skip_val_plot=1):


    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8),
                                           gridspec_kw={'left':0.1, 'right':0.95,
                                                        'bottom':0.1, 'top':0.95})
    ax1, ax2, ax3, ax4 = axes[0, 1], axes[1, 1], axes[0, 0], axes[1, 0]

    # first plot the original 1D power specturm
    ax1.loglog(1/f1d[::skip_val_plot][::-1], p1d[::skip_val_plot][::-1], '.', c='gray', alpha=0.5, markersize=3, Label="Power",
              rasterized=True)
    ax1.loglog(1/bin_center[::-1], bin_med[::-1], 'ko--', markersize=8, alpha=1,
              label="Log bins at median")
    ax1.loglog(1/bin_center[::-1], pl_fit[::-1], 'r-', lw=1, label="Power-law fit to bins")
    # ax1.loglog(bin_center, foo, 'b-', lw=1.5, label="chi2")
    ax1.set_xlim(maxWVL, step)
    ax1.set_xticklabels(["{:,.0f}\n[{:g}]".format(a, 1/a) for a in ax1.get_xticks()])
    ax1.set_xlabel('Wavelength (m) / [Frequency (m$^{-1}$)]', fontsize=12)
    ax1.set_ylabel('Mean Squared Amplitude', fontsize=12)
    # ax1.set_ylim(10**-8, 10**1)
    ax1.grid(True, which="both")
    ax1.legend(loc='lower left', fontsize=12)
    # %
    # then plot the normalized 1D power specturm
    ax2.semilogx(1/f1d[::skip_val_plot][::-1], p1d_norm[::skip_val_plot][::-1], '.', c='gray', alpha=0.5, markersize=8, Label="Power",
              rasterized=True)
    ax2.set_ylim(0, p1d_norm.max())
    ax2.set_xlim(maxWVL, step)
    ax2.set_xticklabels(["{:,.0f}\n[{:g}]".format(a, 1/a) for a in ax2.get_xticks()])
    ax2.set_xlabel('Wavelength (m) / [Frequency (m$^{-1}$)]', fontsize=12)
    ax2.set_ylabel('Normalized Amplitude', fontsize=12)
    ax2.text(1/f1d.max() * 4, p1d_norm.max()/2, 'Power (variance)\nat 1-{} pixels: {:.0f}%'.format(s_wvl, s_power*100),
             bbox=dict(facecolor='w', edgecolor='k', pad=1, alpha=0.8), fontsize=12)
    ax2.grid(True, which="both")
    ax2.legend(loc='lower left', fontsize=12)
    # %
    # now plot the original
    im = ax3.imshow(hphs_, cmap='cividis', interpolation='nearest', vmax=np.percentile(hphs_, 99.9))
    cbar = plt.colorbar(im, ax=ax3, shrink=0.5)
    cbar.set_label('HPHS', rotation=270, labelpad=20, fontsize=12)
    # ax3.imshow(hs, cmap='Greys', alpha=0.1)
    # these next lines add some labels
    ax3.set_xticklabels(["{:.0f}".format(x * step) for x in ax3.get_xticks()])
    ax3.set_yticklabels(["{:.0f}".format(y * step) for y in ax3.get_yticks()])
    ax3.set_xlabel("West-East (m)", fontsize=12)
    ax3.set_ylabel("North-South (m)", fontsize=12)
    ax3.set_title('{}; BLOB {}, {}'.format(dem_name, label, slp_asp_label), fontsize=12)

    # also plot the normalized 2D power spectrum
    im = ax4.imshow(p2d_norm, interpolation='nearest', vmax=np.nanpercentile(p2d_norm, 99.99))
    fig.colorbar(im, ax=ax4, label="Normalized Power (color to 99.99th perc.)")
    ax4.set_xlabel("X Frequency (m$^{-1}$)", fontsize=12)
    ax4.set_ylabel("Y Frequency (m$^{-1}$)", fontsize=12)
    ax4.set_title('P$_{{max}}$ {:.0f}, Orientation {:.0f}$^\circ$, Wvl {:.0f} m'.format(maxP, abs(F_ang1d_maxP), wvl_maxP))
    nfy, nfx = f2d.shape
    nyq = f2d[nfy//2 + 1, 0]
    n_labels = 8
    ax4.set_xticks(np.linspace(1, nfx, n_labels))
    ax4.set_yticks(np.linspace(1, nfy, n_labels))
    ticks = ["{:.3f}".format(a) for a in np.linspace(-nyq, nyq, n_labels)]
    ax4.set_xticklabels(ticks)
    ax4.set_yticklabels(ticks)

    fig.savefig(file_name, dpi=150)
    plt.close()


def plotDFT(file_name, dem_name, xx, yy, f1d, p1d, bin_center, bin_med, pl_fit,
            maxWVL, step, p1d_norm, p2d_norm, hphs_, maxP,
            f2d, F_ang1d_maxP, wvl_maxP, s_wvl, s_power, skip_val_plot=1):


    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8),
                                           gridspec_kw={'left':0.1, 'right':0.95,
                                                        'bottom':0.1, 'top':0.95})
    ax1, ax2, ax3, ax4 = axes[0, 1], axes[1, 1], axes[0, 0], axes[1, 0]

    # first plot the original 1D power specturm
    ax1.loglog(1/f1d[::skip_val_plot][::-1], p1d[::skip_val_plot][::-1], '.', c='gray', alpha=0.5, markersize=3, Label="Power",
              rasterized=True)
    ax1.loglog(1/bin_center[::-1], bin_med[::-1], 'ko--', markersize=8, alpha=1,
              label="Log bins at median")
    ax1.loglog(1/bin_center[::-1], pl_fit[::-1], 'r-', lw=1, label="Power-law fit to bins")
    # ax1.loglog(bin_center, foo, 'b-', lw=1.5, label="chi2")
    ax1.set_xlim(maxWVL, step)
    ax1.set_xticklabels(["{:,.0f}\n[{:g}]".format(a, 1/a) for a in ax1.get_xticks()])
    ax1.set_xlabel('Wavelength (m) / [Frequency (m$^{-1}$)]', fontsize=12)
    ax1.set_ylabel('Mean Squared Amplitude', fontsize=12)
    # ax1.set_ylim(10**-8, 10**1)
    ax1.grid(True, which="both")
    ax1.legend(loc='lower left', fontsize=12)
    # %
    # then plot the normalized 1D power specturm
    ax2.semilogx(1/f1d[::skip_val_plot][::-1], p1d_norm[::skip_val_plot][::-1], '.', c='gray', alpha=0.5, markersize=8, Label="Power",
              rasterized=True)
    ax2.set_ylim(0, p1d_norm.max())
    ax2.set_xlim(maxWVL, step)
    ax2.set_xticklabels(["{:,.0f}\n[{:g}]".format(a, 1/a) for a in ax2.get_xticks()])
    ax2.set_xlabel('Wavelength (m) / [Frequency (m$^{-1}$)]', fontsize=12)
    ax2.set_ylabel('Normalized Amplitude', fontsize=12)
    ax2.text(1/f1d.max() * 4, p1d_norm.max()/2, 'Power (variance)\nat 1-{} pixels: {:.0f}%'.format(s_wvl, s_power*100),
             bbox=dict(facecolor='w', edgecolor='k', pad=1, alpha=0.8), fontsize=12)
    ax2.grid(True, which="both")
    ax2.legend(loc='lower left', fontsize=12)
    # %
    # now plot the original
    im = ax3.imshow(hphs_, cmap='cividis', interpolation='nearest', vmax=np.percentile(hphs_, 99.9))
    cbar = plt.colorbar(im, ax=ax3, shrink=0.5)
    cbar.set_label('HPHS', rotation=270, labelpad=20, fontsize=12)
    # ax3.imshow(hs, cmap='Greys', alpha=0.1)
    # these next lines add some labels
    ax3.set_xticklabels(["{:.0f}".format(x * step) for x in ax3.get_xticks()])
    ax3.set_yticklabels(["{:.0f}".format(y * step) for y in ax3.get_yticks()])
    ax3.set_xlabel("West-East (m)", fontsize=12)
    ax3.set_ylabel("North-South (m)", fontsize=12)
    ax3.set_title('{}; TILE {}, {}'.format(dem_name, xx, yy), fontsize=12)

    # also plot the normalized 2D power spectrum
    im = ax4.imshow(p2d_norm, interpolation='nearest', vmax=np.nanpercentile(p2d_norm, 99.99))
    fig.colorbar(im, ax=ax4, label="Normalized Power (color to 99.99th perc.)")
    ax4.set_xlabel("X Frequency (m$^{-1}$)", fontsize=12)
    ax4.set_ylabel("Y Frequency (m$^{-1}$)", fontsize=12)
    ax4.set_title('P$_{{max}}$ {:.0f}, Orientation {:.0f}$^\circ$, Wvl {:.0f} m'.format(maxP, abs(F_ang1d_maxP), wvl_maxP))
    nfy, nfx = f2d.shape
    nyq = f2d[nfy//2 + 1, 0]
    n_labels = 8
    ax4.set_xticks(np.linspace(1, nfx, n_labels))
    ax4.set_yticks(np.linspace(1, nfy, n_labels))
    ticks = ["{:.3f}".format(a) for a in np.linspace(-nyq, nyq, n_labels)]
    ax4.set_xticklabels(ticks)
    ax4.set_yticklabels(ticks)

    fig.savefig(file_name, dpi=150)
    plt.close()

# %% misc functions

# def winVar(img, wlen):
#   wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen),
#     borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
#   return wsqrmean - wmean*wmean

def window_stdev(arr, radius):
    c1 = ndi.uniform_filter(arr, radius*2, mode='constant', origin=-radius)
    c2 = ndi.uniform_filter(arr*arr, radius*2, mode='constant', origin=-radius)
    return ((c2 - c1*c1)**.5)#[:-radius*2+1,:-radius*2+1]

def sliding_std_dev(image_original,radius=5):
    height, width = image_original.shape
    result = np.zeros_like(image_original) # initialize the output matrix
    hgt = range(radius,height-radius)
    wdt = range(radius,width-radius)
    for i in hgt:
        for j in wdt:
            result[i,j] = np.std(image_original[i-radius:i+radius,j-radius:j+radius])
    return result

def exportlas(fn, var, pts):
    import laspy
    from matplotlib.cm import magma as cmap
    v = var - np.min(var)
    v /= v.max()
    rgb = cmap(v)
    rgb = rgb[:, :3]
    rgb *= 65535
    rgb = rgb.astype('uint')
    header = laspy.header.Header()
    header.data_format_id = 2
    f = laspy.file.File(fn, mode = 'w', header = header)
    f.header.scale = [0.001, 0.001, 0.001]
    f.header.offset = [pts[:,0].min(), pts[:,1].min(), pts[:,2].min()]
    f.x = pts[:, 0]
    f.y = pts[:, 1]
    f.z = pts[:, 2]
    if pts.shape[1] == 4:
        f.intensity = pts[:, 3]
    f.set_red(rgb[:, 0])
    f.set_green(rgb[:, 1])
    f.set_blue(rgb[:, 2])
    f.close()

def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code
    https://stackoverflow.com/a/40140326/4556479"""
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code

# %%

def loadDEMclipForResampling(dem, xx, yy, tile_size_px):
    kwargs = {'format' : 'VRT',
              'srcWin' : [xx, yy, tile_size_px, tile_size_px]}

    ds = gdal.Translate('', dem, **kwargs)
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    nan = band.GetNoDataValue()
    # print(nan)

    # read as array and set NaN
    el = band.ReadAsArray().astype(float)
    el[el == nan] = np.nan
    # print(el.shape)

    # ONLY IF COMPARING FULL TILE:
    # if the shape ends in 1, then the tile had one east column and
    # one south row removed:
        # https://forum.sentinel-hub.com/t/copernicus-dem-data-available-on-aws/3027/2
        # https://copernicus-dem-30m.s3.amazonaws.com/readme.html
    # if el.shape[0] % 10 != 0:
    #     # print('yup')
    #     el = el[0:-1, 0:-1]
    #     # print(el.shape)

    # get pixel size
    cols = el.shape[1]
    rows = el.shape[0]
    minx, maxy = gt[0], gt[3]
    maxx, miny = gt[0] + gt[1] * cols, gt[3] + gt[5] * rows

    # read crs
    crs = CRS.from_wkt(ds.GetProjection()).to_epsg()

    # get step in m if geographic projection
    if crs == 4326:
        print('converting to UTM coords')
        epsg_code = convert_wgs_to_utm(minx, miny)
        pp = Proj('EPSG:{}'.format(epsg_code))
        proj = CRS.from_epsg(epsg_code).to_wkt()
        minx, miny = pp(minx, miny)
        maxx, maxy = pp(maxx, maxy)
    psx = (maxx - minx) / cols
    psy = (maxy - miny) / rows
    step = np.round((psx + psy) / 2, 0)

    minLon, maxLon, minLat, maxLat = minx, maxx, miny, maxy
    pixel_size = step

    # pixel_size = (((maxLon - minLon) / cols) + ((maxLat - minLat) / rows)) / 2
    Lats = np.arange(minLat + (pixel_size / 2), maxLat, pixel_size)[::-1]
    Lons = np.arange(minLon + (pixel_size / 2), maxLon, pixel_size)
    gridLon, gridLat = np.meshgrid(Lons, Lats)

    # close dataset
    ds = None

    return el, step, minLon, maxLon, minLat, maxLat, gridLon, gridLat

def loadDEMclipForCartopy(dem, xx, yy, tile_size_px):
    kwargs = {'format' : 'VRT',
              'srcWin' : [xx, yy, tile_size_px, tile_size_px]}

    ds = gdal.Translate('', dem, **kwargs)
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    nan = band.GetNoDataValue()
    # print(nan)

    # read as array and set NaN
    el = band.ReadAsArray().astype(float)
    el[el == nan] = np.nan
    # print(el.shape)

    # ONLY IF COMPARING FULL TILE:
    # if the shape ends in 1, then the tile had one east column and
    # one south row removed:
        # https://forum.sentinel-hub.com/t/copernicus-dem-data-available-on-aws/3027/2
        # https://copernicus-dem-30m.s3.amazonaws.com/readme.html
    # if el.shape[0] % 10 != 0:
    #     # print('yup')
    #     el = el[0:-1, 0:-1]
    #     # print(el.shape)

    # get pixel size
    cols = el.shape[1]
    rows = el.shape[0]
    minx, maxy = gt[0], gt[3]
    maxx, miny = gt[0] + gt[1] * cols, gt[3] + gt[5] * rows

    minLon, maxLon, minLat, maxLat = minx, maxx, miny, maxy

    pixel_size = (((maxLon - minLon) / cols) + ((maxLat - minLat) / rows)) / 2
    Lats = np.arange(minLat + (pixel_size / 2), maxLat, pixel_size)[::-1]
    Lons = np.arange(minLon + (pixel_size / 2), maxLon, pixel_size)
    gridLon, gridLat = np.meshgrid(Lons, Lats)

    # read crs
    crs = CRS.from_wkt(ds.GetProjection()).to_epsg()

    # get step in m if geographic projection
    if crs == 4326:
        epsg_code = convert_wgs_to_utm((maxx + minx)/2, (maxy + miny)/2)
        pp = Proj('EPSG:{}'.format(epsg_code))
        proj = CRS.from_epsg(epsg_code).to_wkt()
        minx, miny = pp(minx, miny)
        maxx, maxy = pp(maxx, maxy)
    psx = (maxx - minx) / cols
    psy = (maxy - miny) / rows
    step = np.round((psx + psy) / 2, 0)

    # close dataset
    ds = None

    return el, psx, psy, minLon, maxLon, minLat, maxLat, gridLon, gridLat





# %%
def maxSquareRC(M):
    """ expects a binary array of 0 and 1
    https://www.geeksforgeeks.org/maximum-size-sub-matrix-with-all-1s-in-a-binary-matrix/"""
    R = len(M) # no. of rows in M[][]
    C = len(M[0]) # no. of columns in M[][]

    S = [[0 for k in range(C)] for l in range(R)]
    # here we have set the first row and column of S[][]

    # Construct other entries
    for r in range(1, R):
        for c in range(1, C):
            if (M[r][c] == 1):
                S[r][c] = min(S[r][c-1], S[r-1][c],
                              S[r-1][c-1]) + 1
            else:
                S[r][c] = 0

    # Find the maximum entry and
    # indices of maximum entry in S[][]
    max_of_s = S[0][0]
    max_r = 0
    max_c = 0
    for r in range(R):
        for c in range(C):
            if (max_of_s < S[r][c]):
                max_of_s = S[r][c]
                max_r = r
                max_c = c

    r0, r1, c0, c1 = max_r - max_of_s + 1, max_r, max_c - max_of_s + 1, max_c

    return r0, r1, c0, c1




# %% functions for hillshadeing

# def horn_slope(dem, xx, yy, tile_size_px):
#     kwargs = {'format' : 'VRT',
#               'srcWin' : [xx, yy, tile_size_px, tile_size_px]}

#     ds = gdal.Translate('', dem, **kwargs)
#     band = ds.GetRasterBand(1)
#     gt = ds.GetGeoTransform()
#     nan = band.GetNoDataValue()
#     # print(nan)

#     # read as array and set NaN
#     el = band.ReadAsArray().astype(float)



def clip_dem(dem, dem_name, clipper):

    # open the full dataset to pull out some metadata

    # open raster
    ds = gdal.Open(dem)
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    step = gt[1]

    # create an in memory clip of the data for comparison purposes

    kwargs = {'dstSRS' : 'EPSG:32719', 'xRes' : step, 'yRes' : step,
              'resampleAlg' : 'bilinear', 'format' : 'VRT', 'dstNodata' : -9999,
              'cutlineDSName' : clipper, 'cropToCutline' : 'True', 'targetAlignedPixels' : 'True'}

    ds = gdal.Warp('', dem, **kwargs)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    band = ds.GetRasterBand(1)

    # get the NaN value
    nan = band.GetNoDataValue()
    if nan == None:
        nan = 0
    print(nan)

    # read as array and set NaN
    el = band.ReadAsArray().astype(float)
    el[el == nan] = np.nan

    # remove nan rows / columns
    el = el[~np.isnan(el).all(axis=1)]
    el = el[:, ~np.isnan(el).all(axis=0)]
    el = el[~np.isnan(el).any(axis=1)]

    return el, step, nan, gt, proj


def pyshedsDA_raster(dem):
    """
    https://mattbartos.com/pysheds/accumulation.html
    """

    # load the grid object
    grid = Grid.from_raster(dem, data_name='dem')

    # Fill depressions in DEM
    grid.fill_depressions('dem', out_name='flooded_dem')

    # Resolve flats in DEM
    grid.resolve_flats('flooded_dem', out_name='inflated_dem')

    # Specify directional mapping
    # See here: https://mattbartos.com/pysheds/flow-directions.html
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Compute flow directions
    # -------------------------------------
    grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap, routing='d8')

    # Calculate flow accumulation
    # --------------------------
    areas = grid.cell_area(inplace=False)
    grid.accumulation(data='dir', dirmap=dirmap, out_name='acc', weights=areas)
    flowacc = grid.view('acc')

    return flowacc


def pyshedsDA_array(el, nan, gt, proj):
    """
    https://mattbartos.com/pysheds/accumulation.html
    """
    # get the projection as Proj4
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    proj4 = srs.ExportToProj4()
    crs = Proj(proj4, preserve_units=True)

    # set the affine transform
    affine = rasterio.Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])

    # load the grid object
    grid = Grid()
    grid.add_gridded_data(data=el, data_name='dem',
                              shape=el.shape,
                              affine=affine,
                              crs=crs,
                              nodata=nan,
                              mask=None)

    # Fill depressions in DEM
    grid.fill_depressions('dem', out_name='flooded_dem')

    # Resolve flats in DEM
    grid.resolve_flats('flooded_dem', out_name='inflated_dem')

    # Specify directional mapping
    # See here: https://mattbartos.com/pysheds/flow-directions.html
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Compute flow directions
    # -------------------------------------
    grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap, routing='d8')

    # Calculate flow accumulation
    # --------------------------
    areas = grid.cell_area(inplace=False)
    grid.accumulation(data='dir', dirmap=dirmap, out_name='acc', weights=areas)
    flowacc = grid.view('acc')

    return flowacc


def pyshedsCatchment_array(el, nan, gt, proj, lon, lat):
    """
    https://mattbartos.com/pysheds/accumulation.html
    """
    # get the projection as Proj4
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    proj4 = srs.ExportToProj4()
    crs = Proj(proj4, preserve_units=True)

    # set the affine transform
    affine = rasterio.Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])

    # load the grid object
    grid = Grid()
    grid.add_gridded_data(data=el, data_name='dem',
                              shape=el.shape,
                              affine=affine,
                              crs=crs,
                              nodata=nan,
                              mask=None)

    # Fill depressions in DEM
    grid.fill_depressions('dem', out_name='flooded_dem')

    # Resolve flats in DEM
    grid.resolve_flats('flooded_dem', out_name='inflated_dem')

    # Specify directional mapping
    # See here: https://mattbartos.com/pysheds/flow-directions.html
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Compute flow directions
    # -------------------------------------
    grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap, routing='d8')

    # Calculate flow accumulation
    # --------------------------
    areas = grid.cell_area(inplace=False)
    grid.accumulation(data='dir', dirmap=dirmap, out_name='acc', weights=areas)
    flowacc = grid.view('acc')

    # Delineate catchment
    # --------------------------
    grid.catchment(data='dir', x=lon, y=lat,
                   out_name='catch',
                   recursionlimit=15000, xytype='label')
    catch = grid.view('catch')

    return catch


def smoothed_hs_calc(el, step, azimuths=(0, 90, 180, 270), angle_altitudes=(15, 45, 75), sigma=0.5, filt_w=3):

    hillshades = np.zeros((el.shape[0], el.shape[1], len(azimuths), len(angle_altitudes)))
    for ang_num, ang in enumerate(angle_altitudes):
        for az_num, az in enumerate(azimuths):
            hs = hillshade(el, step, azimuth=az, angle_altitude=ang)
            hillshades[:,:,az_num,ang_num] = hs[:,:]

    # also generate "smoothed" hillshades
    t = (((filt_w - 1)/2)-0.5)/sigma # https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
    el_sm = ndi.gaussian_filter(el, sigma, truncate=t)
    hillshades_sm = np.zeros((el.shape[0], el.shape[1], len(azimuths), len(angle_altitudes)))
    for ang_num, ang in enumerate(angle_altitudes):
        for az_num, az in enumerate(azimuths):
            hs = hillshade(el_sm, step, azimuth=az, angle_altitude=ang)
            hillshades_sm[:,:,az_num,ang_num] = hs[:,:]

    return hillshades, hillshades_sm

# %% poly fit function




