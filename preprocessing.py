# Title: Script for NOAA and Landsat-1 preprocessing
# Author: Elias Frey
# Date: 15.04.23
# Description: This script processes NOAA and Landsat-1 scenes;
#              NOAA: Convert to TIF and reassign proper NOAA Polar Stereographic Projection, output as TIF
#              Landsat: Re-projection into NOAA Polar Stereographic Projection, output as TIF

import os
from pathlib import PurePath
import rasterio
from rasterio.crs import CRS
import rioxarray
import pyproj
from pyproj import CRS
########################################################################################################################
# USER INPUT: #

# Choose satellites to process
process_noaa = True  # True: noaa TIFs will be created; False: no noaa processing
process_landsat = True  # True: landsat re-projection will be created; False: no landsat processing

# Adjust path to data
data_path = 'C:\\Users\\efrey\\Desktop\\TEST_PATH'

########################################################################################################################
noaa_data_path = os.path.join(data_path, 'NOAA')
landsat_data_path = os.path.join(data_path, 'Landsat')


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


def reproject_landsat(input_landsat):
    """ Reproject Landsat-1 to NOAA Polar Stereographic """
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


def convert_to_tif(input_noaa):
    """ Convert NOAA netCDF to raster TIF file """
    if 'VIS' in os.path.basename(input_noaa):
        # Open and select index where "vis_norm_remapped" is located and drop "band" dimension
        noaa_nc = rioxarray.open_rasterio(input_noaa, mask_and_scale=True)[2].drop_dims('band')
        noaa_crs = CRS.from_cf(noaa_nc.crs.attrs)  # Extract CRS with pyproj library
        noaa_nc = noaa_nc.rio.write_crs(noaa_crs.to_string(), inplace=True)  # Assign found CRS to NOAA file
        noaa_nc['vis_norm_remapped'] = noaa_nc['vis_norm_remapped'].rio.write_nodata(255, inplace=True)  # assign nodata
        noaa_nc.attrs['valid_max'] = 255  # Correct noaa max valid attribute
        assert noaa_nc['vis_norm_remapped'].rio.nodata == 255
        # Mask NOAA VIS band with flags, set nan values to 255 and change dtype to uint8
        noaa_vis = noaa_nc['vis_norm_remapped'].where(noaa_nc['flag_remapped'] == 0).fillna(255).astype(dtype='uint8')
        output_noaa = os.path.join(create_dir(input_noaa, 'noaa'), os.path.basename(input_noaa)[:-4] + "_Conv.TIF")
        noaa_vis.rio.to_raster(output_noaa)  # Convert and save NOAA netCDF file into TIF raster file

    # @ TODO same for IR
    elif 'IRday' in os.path.basename(input_noaa):
        # Open and select index where "calibrated_longwave_flux" is located and drop "band" dimension
        noaa_nc = rioxarray.open_rasterio(input_noaa)[1].drop_dims('band')
        noaa_crs = CRS.from_cf(noaa_nc.crs.attrs)  # Extract CRS with pyproj library
        noaa_nc = noaa_nc.rio.write_crs(noaa_crs.to_string(), inplace=True)  # Assign found CRS to NOAA file
        noaa_nc['calibrated_longwave_flux'] = noaa_nc['calibrated_longwave_flux'].rio.write_nodata(32767, inplace=True)  # assign nodata
        noaa_nc.attrs['valid_max'] = 20000  # Correct noaa max valid attribute
        assert noaa_nc['calibrated_longwave_flux'].rio.nodata == 32767
        # Mask NOAA IR band with flags, set nan values to 255 and change dtype to uint8
        noaa_ir = noaa_nc['calibrated_longwave_flux'].where(noaa_nc['flag_remapped'] == 0).fillna(32767).astype(dtype='int16')
        output_noaa = os.path.join(create_dir(input_noaa, 'noaa'), os.path.basename(input_noaa)[:-4] + "_Conv.TIF")
        noaa_ir.rio.to_raster(output_noaa)  # Convert and save NOAA netCDF file into TIF raster file
    else:
        print(f'Error in function "convert_to_tif": VIS or IRday not found in filename: {os.path.basename(input_noaa)}')


def noaa_processing():
    """ Main NOAA processing function """
    # Check and list folders
    root, dirs, files = next(os.walk(os.path.join(noaa_data_path, 'input', 'film'), topdown=True))
    folders = [os.path.join(root, d) for d in dirs]
    for folder in folders:
        # Check and list files
        print(folder)
        root, dirs, files = next(os.walk(folder, topdown=True))
        scenes = [os.path.join(root, s) for s in files if '.nc' in s]
        for scene in scenes:
            print(scene)
            # Apply noaa converting function on each scene
            convert_to_tif(scene)
            print(f'NOAA scene {scene} successfully converted to netCDF')
    print(f'NOAA processing finished!')


def landsat_processing():
    """ Main Landsat processing function """
    # Check and list folders
    root, dirs, files = next(os.walk(os.path.join(landsat_data_path, 'input'), topdown=True))
    folders = [os.path.join(root, d) for d in dirs]
    for folder in folders:
        # Check and list files
        print(folder)
        root, dirs, files = next(os.walk(folder, topdown=True))
        scenes = [os.path.join(root, s) for s in files if '.TIF' in s]
        scenes[0], scenes[5] = scenes[5], scenes[0]
        print(scenes)
        for scene in scenes:
            print(scene)
            # Apply landsat re-projecting function on each scene
            reproject_landsat(scene)
            # Mask only band data
            if 'QA' not in scene:
                # Apply landsat QA (reprojected) masking function on each scene
                mask_landsat(scene)
            # create_dir(scene, 'landsat')
            print(f'Landsat-1 scene {scene} successfully converted to netCDF')
    print('Landsat-1 processing finished!')


if __name__ == '__main__':
    if process_noaa and process_landsat:
        noaa_processing()
        landsat_processing()
    elif process_noaa and not process_landsat:
        noaa_processing()
    elif process_landsat and not process_noaa:
        landsat_processing()
    else:
        print(f'Nothing to process with: process_noaa = {process_noaa} and process_landsat = {process_landsat} ...')
