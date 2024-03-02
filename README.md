# Preprocess NOAA and Landsat-1 scenes
NOAA: Very High Resolution Radiometer (VHRR) sensor providing VIS (0.52-0.72 μm) and IR (10.5-12.5 μm) bands with resampled resolution of 10,193.8m.  
Landsat-1: Multispectral Scanner (MSS) sensor providing four bands (green, red, NIR1, NIR2) with 80m resolution.

This script pre-processes NOAA and Landsat-1 scenes by applying QA masks, reproject to same projection (Polar stereographic), align pixel values and convert to Tagged Image File Format (.tif) depending on the scene/data type;
NOAA: Convert to TIF and reassign proper NOAA Polar Stereographic Projection, output as .TIF file
Landsat: Re-project into NOAA Polar Stereographic Projection, output as .TIF file

#### Files:
##### preprocessing.py: NOAA and Landsat-1 preprocessing (CRS assignment, reprojection, masking)
##### preprocessing_environment.yml: Conda environment for NOAA and Landsat-1 preprocessing
