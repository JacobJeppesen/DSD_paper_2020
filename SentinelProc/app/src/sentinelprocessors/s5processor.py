import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from absl import flags, logging
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

from src.sentinelprocessors.base import BaseProcessor

FLAGS = flags.FLAGS


class S5Processor(BaseProcessor):
    def __init__(self, products_df, directory, compress_gtiff=True, overwrite_products=False):
        super().__init__(products_df, directory, compress_gtiff=compress_gtiff, overwrite_products=overwrite_products)
        self.compress_gtiff = FLAGS.compress_gtiff
        self.overwrite_products = FLAGS.overwrite

    def process(self, index):
        # Extract the row from the dataframe with the product process (use copy to avoid "copy of a slice warning")
        # https://stackoverflow.com/questions/31468176/setting-values-on-a-copy-of-a-slice-from-a-dataframe
        product = self.products_df.iloc[[index]].copy()
        # TODO: Should the .nc file really be moved? or should it stay in zipfiles folder and only the output gtiff file
        #       should be in the output_data folder?
        # self.__move_product(product)
        self.__convert_nc_to_basemap(product, map_type='DK')
        self.__convert_nc_to_basemap(product, map_type='EU')

    def __move_product(self, product):
        if not product['product_path'].values[0].exists() or self.overwrite_products:
            file_name = product['title'].values[0]
            src_path = (self.directory / 'zipfiles' / file_name).with_suffix('.nc')
            dst_path = (product['product_path'].values[0] / file_name).with_suffix('.nc')
            if not dst_path.parent.exists():  # Create directory if it does not exist
                os.makedirs(dst_path.parent)
            shutil.copy(str(src_path), str(dst_path))
            logging.info('Product has been moved: ' + product['title'].values[0])
        else:
            logging.info('Product already exists in output folder: ' + product['title'].values[0])

    def __convert_nc_to_basemap(self, product, map_type='EU'):
        file_name = product['title'].values[0]
        begin_time = product['beginposition'].values[0]
        end_time = product['endposition'].values[0]

        src_path = (self.directory / 'zipfiles' / file_name).with_suffix('.nc')

        # See here for list of available variables:
        # https://earth.esa.int/web/sentinel/technical-guides/sentinel-5p/level-2/products/main-variables
        producttype_variable_mapping = {'Ozone': ['ozone_total_vertical_column', 1e-1, 2e-1],
                                        'Sulphur Dioxide': ['sulfurdioxide_total_vertical_column', 1e-9, 1e-1],
                                        'Nitrogen Dioxide': ['nitrogendioxide_tropospheric_column', 1e-13, 1e-1],
                                        'Methane': ['methane_mixing_ratio', 1.5e3, 2.2e3],
                                        'Formaldehyde': ['formaldehyde_tropospheric_vertical_column', 1e-9, 1e-2],
                                        'Carbon Monoxide': ['carbonmonoxide_total_column', 1e-2, 1e-1],
                                        'Aerosol Index': ['aerosol_index_354_388', 1e-5, 1e1],
                                        'Aerosol Layer Height': ['aerosol_mid_pressure', 1e4, 1e5],
                                        'Cloud': ['cloud_fraction', 1e-2, 1e0]}
        producttype = product['producttypedescription'].values[0]
        variable_name = producttype_variable_mapping[producttype][0]
        min_value = producttype_variable_mapping[producttype][1]
        max_value = producttype_variable_mapping[producttype][2]

        ncdata = Dataset(src_path, mode='r')
        lon = ncdata.groups['PRODUCT'].variables['longitude'][:][0, :, :]
        lat = ncdata.groups['PRODUCT'].variables['latitude'][:][0, :, :]
        values = ncdata.groups['PRODUCT'].variables[variable_name][0, :, :]
        value_units = ncdata.groups['PRODUCT'].variables[variable_name].units

        plt.figure(figsize=(20, 20))
        plt.rcParams.update({'font.size': 18})
        if map_type == 'DK':
            lat_0, lon_0, map_width, map_height, basemap_resolution = 56, 10, 500000, 500000, 'i'
            dst_path = (product['product_path'].values[0] / 'basemaps' / 'DK' /
                        product['product_type'].values[0] / file_name).with_suffix('.png')
            if not dst_path.parent.exists():  # Create directory if it does not exist
                os.makedirs(dst_path.parent)
        elif map_type == 'EU':
            lat_0, lon_0, map_width, map_height, basemap_resolution = 53, 10, 4500000, 4500000, 'l'
            dst_path = (product['product_path'].values[0] / 'basemaps' / 'EU' /
                        product['product_type'].values[0] / file_name).with_suffix('.png')
            if not dst_path.parent.exists():  # Create directory if it does not exist
                os.makedirs(dst_path.parent)
        else:
            logging.fatal("map_type argument in __convert_nc_to_basemap() must be either 'EU' or 'DK'")

        m = Basemap(width=map_width, height=map_height, resolution=basemap_resolution, projection='stere',
                    lat_ts=25, lat_0=lat_0, lon_0=lon_0)
        xi, yi = m(lon, lat)

        # Plot Data
        cs = m.pcolor(xi, yi, np.squeeze(values), norm=LogNorm(), vmin=min_value, vmax=max_value, cmap='jet')

        # Add Grid Lines
        # m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
        # m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10)

        # Add Coastlines, States, and Country Boundaries
        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()

        # Add Colorbar
        cbar = m.colorbar(cs, location='bottom', pad="4%")
        cbar.set_label(value_units)

        # Add Title
        plt.title(producttype + " (" + variable_name + ")" +
                  "\n Begin time (entire product): " + str(begin_time)[0:10] + " " + str(begin_time)[11:19] +
                  "\n End time (entire product):    " + str(end_time)[:10] + " " + str(end_time)[11:19])
        plt.tight_layout()
        plt.savefig(dst_path)

        plt.close()


        # From https://gis.stackexchange.com/questions/261677/determine-the-geotransformation-to-convert-a-netcdf-to-geotiff
        # xmin, ymin, xmax, ymax = [lon.min(), lat.min(), lon.max(), lat.max()]
        # lon = ((lon + 180) % 360) - 180
        #
        # nx = len(lon)
        # ny = len(lat)
        # xmin, ymin, xmax, ymax = [lon.min(), lat.min(), lon.max(), lat.max()]
        # xres = (xmax - xmin) / float(nx)
        # yres = (ymax - ymin) / float(ny)
        # geotransform = (xmin, xres, 0, ymax, 0, -yres)
        #
        # dst_ds = gdal.GetDriverByName('GTiff').Create('/data/output.tif', ny, nx, 1, gdal.GDT_Float32)
        #
        # dst_ds.SetGeoTransform(geotransform)  # specify coords
        # srs = osr.SpatialReference()  # establish encoding
        # srs.ImportFromEPSG(3857)  # WGS84 lat/long
        # dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
        # dst_ds.GetRasterBand(1).WriteArray(no2)  # write r-band to the raster
        # dst_ds.FlushCache()  # write to disk
        # dst_ds = None




