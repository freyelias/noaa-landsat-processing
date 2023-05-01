# Title: Script to preprocess NOAA and Landsat-1 scenes
# Author: Elias Frey
# Date: 27.04.23
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 RSGB/UniBe (Remote Sensing Research Group, University of Bern)
#
#
# Description: This script processes NOAA and Landsat-1 scenes;
#              NOAA: Convert to TIF and reassign proper NOAA Polar Stereographic Projection, output as .TIF file
#              Landsat: Re-project into NOAA Polar Stereographic Projection, output as .TIF file
#
#
# USER INFORMATION
#
# File:                 preprocessing.py
# Synopsis:             python preprocessing.py process_noaa[yes/no] process_landsat[yes/no] main_data_path[C:\Users\..]
#                       Example:
#                       python preprocessing.py yes yes C:\Users\efrey\Desktop\main_folder
#
# Folder structure:    Main folder\NOAA\input\film\"NOAA year folders"
#                      Main folder\Landsat\input\"Landsat scene folders"
#                      Example:
#                      C:\Users\efrey\Desktop\data\NOAA\input\film\1973_03
#                      C:\Users\efrey\Desktop\data\Landsat\input\LM01_L1TP_037022_19730130_20200909_02_T2


""" Preprocess NOAA and Landsat-1 imagery """

import os
import sys
import datetime
import time
from pathlib import PurePath, Path
import rasterio
from rasterio.crs import CRS
import warnings
import rioxarray
import pyproj
from pyproj import CRS


def create_dir(input_scene, satellite):
    """ Create output folder structure """
    if satellite == 'noaa':
        out_dir_noaa = os.path.join(noaa_data_path, 'output', 'film', PurePath(input_scene).name.split('.')[5] + '_' +
                                    PurePath(input_scene).name.split('.')[6])
        if not os.path.exists(out_dir_noaa):
            os.makedirs(out_dir_noaa)
        return out_dir_noaa

    elif satellite == 'landsat':
        out_dir_landsat = os.path.join(os.path.join(landsat_data_path, 'output', PurePath(input_scene).name[0:40]))
        if not os.path.exists(out_dir_landsat):
            os.makedirs(out_dir_landsat)
        return out_dir_landsat

    else:
        print('Satellite type unknown.. did you mean noaa or landsat?')


def convert_to_tif(input_noaa):
    """ Convert NOAA netCDF to raster TIF file """
    if 'VIS' in os.path.basename(input_noaa):
        # Ignore rasterio NotGeoreferencedWarning when opening raw NOAA netCDF file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Open and select index where "vis_norm_remapped" is located and drop "band" dimension
            noaa_nc = rioxarray.open_rasterio(input_noaa)[2].drop_dims('band')
        noaa_crs = CRS.from_cf(noaa_nc.crs.attrs)  # Extract CRS with pyproj library
        noaa_nc = noaa_nc.rio.write_crs(noaa_crs.to_string(), inplace=True)  # Assign found CRS to NOAA file
        noaa_nc['vis_norm_remapped'].attrs['valid_min'] = 1  # Noaa max valid attribute
        noaa_nc['vis_norm_remapped'].attrs['valid_max'] = 255  # Noaa max valid attribute
        # Mask NOAA VIS band with flags, set nan values to 255 and change dtype to uint8
        noaa_nc['vis_norm_remapped'] = noaa_nc['vis_norm_remapped'].where(
            noaa_nc['flag_remapped'] == 0).fillna(0).astype(dtype='uint8')
        noaa_nc['vis_norm_remapped'] = noaa_nc['vis_norm_remapped'].rio.write_nodata(0, inplace=True)  # assign nodata
        noaa_nc = noaa_nc.drop_vars('flag_remapped')
        output_noaa = os.path.join(create_dir(input_noaa, 'noaa'), os.path.basename(input_noaa)[:-3] + "_Conv.TIF")
        noaa_nc.isel(time=0).rio.to_raster(output_noaa)  # Convert and save NOAA netCDF file into TIF raster file

    elif 'IRday' in os.path.basename(input_noaa):
        # Ignore rasterio NotGeoreferencedWarning when opening raw NOAA netCDF file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Open and select index where "calibrated_longwave_flux" is located and drop "band" dimension
            noaa_nc = rioxarray.open_rasterio(input_noaa)[1].drop_dims('band')
        noaa_crs = CRS.from_cf(noaa_nc.crs.attrs)  # Extract CRS with pyproj library
        noaa_nc = noaa_nc.rio.write_crs(noaa_crs.to_string(), inplace=True)  # Assign found CRS to NOAA file
        noaa_nc['calibrated_longwave_flux'].attrs['valid_min'] = 0  # Noaa min valid attribute
        noaa_nc['calibrated_longwave_flux'].attrs['valid_max'] = 2000  # Noaa max valid attribute
        # Mask NOAA IR band with flags, set nan values to 255 and change dtype to uint8
        noaa_nc['calibrated_longwave_flux'] = noaa_nc['calibrated_longwave_flux'].where(
            noaa_nc['flag_remapped'] == 0).fillna(32767).astype(dtype='uint16')
        noaa_nc['calibrated_longwave_flux'] = noaa_nc['calibrated_longwave_flux'].rio.write_nodata(32767, inplace=True)
        noaa_nc = noaa_nc.drop_vars(['flag_remapped', 'OLR_longwave_flux', 'IR_count_remapped'])
        output_noaa = os.path.join(create_dir(input_noaa, 'noaa'), os.path.basename(input_noaa)[:-3] + "_Conv.TIF")
        noaa_nc.isel(time=0).rio.to_raster(output_noaa)  # Convert and save NOAA netCDF file into TIF raster file
    else:
        print(f'Error in function "convert_to_tif": VIS or IRday not found in filename: {os.path.basename(input_noaa)}')


def reproject_landsat(input_landsat):
    """ Reproject Landsat-1 to NOAA Polar Stereographic with proj4string """
    # Create a pyproj CRS object from proj4string (proj4string according to NOAA documentation)
    p_crs = pyproj.CRS("+proj=stere +lat_0=90 +lon_0=-80.0 +lat_ts=90 +x_0=0 +y_0=0 +ellps=sphere +units=m +R=6371128")
    noaa_crs = CRS.from_wkt(p_crs.to_wkt())

    # Set environment parameters
    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_USE_HEAD=False,
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="TIF",
    )

    # Load Landsat-1 raster with rio virtual warping method (memory efficient)
    with env:
        with rasterio.open(input_landsat) as src:
            src_crs = src.crs.to_dict()  # Load Landsat-1 original CRS
            with rasterio.vrt.WarpedVRT(src, crs=src_crs) as vrt:
                rds = rioxarray.open_rasterio(vrt)
                rds = rds.rio.reproject(dst_crs=noaa_crs)
                time_str = os.path.basename(input_landsat).split('_')[3]  # Gets LS conv. date from file (e.g. 19730101)
                time_str = str(time_str[0:4] + '-' + time_str[4:6] + '-' + time_str[6:8])  # Assemble to common dt form.
                # Convert to UNIX time format
                ls_time = int(time.mktime(datetime.datetime.strptime(time_str, '%Y-%m-%d').timetuple()))
                rds.attrs['time'] = ls_time  # Assign UNIX time information to raster attribute
                rds.attrs['valid_min'] = 1  # Noaa max valid attribute
                rds.attrs['valid_max'] = 255  # Noaa max valid attribute
                rds = rds.rio.write_nodata(0, inplace=True)  # Assign nodata
                output_landsat = os.path.join(create_dir(input_landsat, 'landsat'), os.path.basename(input_landsat)[:-4]
                                              + "_REPROJ.TIF")
            rds.rio.to_raster(output_landsat)


def mask_landsat(input_landsat):
    """ Mask Landsat-1 scenes with QA (RADSAT) flags """
    # Get reprojected Landsat-1 scene
    ls_scene = rioxarray.open_rasterio(
        os.path.join(create_dir(input_landsat, 'landsat'), os.path.basename(input_landsat)[:-4] + "_REPROJ.TIF"))
    # Get reprojected Landsat QA_RADSAT flag
    ls_mask_qa_name = os.path.join(
        create_dir(input_landsat, 'landsat'), os.path.basename(input_landsat)[:-6] + 'QA_RADSAT_REPROJ.TIF')
    ls_mask_qa = rioxarray.open_rasterio(ls_mask_qa_name)
    # Filter Landsat-1 scene where QA equals 0 and fill nan values with 0 to be dtype 'uint8' compatible
    ls_masked = ls_scene.where(ls_mask_qa == 0).fillna(0).astype(dtype='uint8')
    output_landsat = os.path.join(
        create_dir(input_landsat, 'landsat'), os.path.basename(input_landsat)[:-4] + "_MASKED.TIF")
    ls_masked.rio.to_raster(output_landsat)


def noaa_processing():
    """ Main NOAA processing function """
    # Check and list folders
    root, dirs, files = next(os.walk(os.path.join(noaa_data_path, 'input', 'film'), topdown=True))
    folders = [os.path.join(root, d) for d in dirs]
    for folder in folders:
        # Check and list files
        root, dirs, files = next(os.walk(folder, topdown=True))
        scenes = [os.path.join(root, s) for s in files if '.nc' in s]
        for scene in scenes:
            print(f'Processing NOAA scene:')
            print(f'{os.path.basename(scene)}')
            # Apply noaa converting function on each scene
            convert_to_tif(scene)
            print(f'Successfully finished scene!')
    print(f'NOAA processing finished!')


def landsat_processing():
    """ Main Landsat processing function """
    # Check and list folders
    root, dirs, files = next(os.walk(os.path.join(landsat_data_path, 'input'), topdown=True))
    folders = [os.path.join(root, d) for d in dirs]
    for folder in folders:
        # Check and list files
        root, dirs, files = next(os.walk(folder, topdown=True))
        scenes = [os.path.join(root, s) for s in files if '.TIF' in s]
        scenes[0], scenes[5] = scenes[5], scenes[0]  # Only needed for windwos, Mac has other sorting technique!
        for scene in scenes:
            print(f'Processing Landsat-1 scene:')
            print(f'{os.path.basename(scene)}')
            # Apply landsat re-projecting function on each scene
            reproject_landsat(scene)
            # Mask only band data
            if 'QA' not in scene:
                # Apply landsat QA (reprojected) masking function on each scene
                mask_landsat(scene)
            # create_dir(scene, 'landsat')
            print(f'Successfully finished scene!')
    print('Landsat-1 processing finished!')


if __name__ == '__main__':
    try:
        # Check if user input is correct (synopsis in description)
        print('Pre-processing script for NOAA and Landsat-1 scenes.')
        process_noaa = sys.argv[1]
        process_landsat = sys.argv[2]
        data_path = Path(sys.argv[3])
        # Define NOAA and Landsat paths
        noaa_data_path = os.path.join(data_path, 'NOAA')
        landsat_data_path = os.path.join(data_path, 'Landsat')

    except IndexError:
        print('INPUT ERROR: process_NOAA, process_landsat and main data path as commandline arguments')
        sys.exit(1)

    # ... and suppress errors
    #gdal.PushErrorHandler('CPLQuietErrorHandler')
    #log = logging.getLogger()
    #log.setLevel(logging.DEBUG)

    if 'y' in process_noaa and 'y' in process_landsat:
        noaa_processing()
        landsat_processing()
    elif 'y' in process_noaa and 'n' in process_landsat:
        noaa_processing()
    elif 'y' in process_landsat and 'n' in process_noaa:
        landsat_processing()
    else:
        print(f'Nothing to process with: process_noaa = {process_noaa} and process_landsat = {process_landsat} ... '
              f'Input ["yes" or "y"] for yes or ["no" or "n"] for no.')

    print('Script done!')
