import os
import zipfile

from pathlib import Path
from absl import logging
from osgeo import gdal


class BaseProcessor(object):
    def __init__(self, products_df, directory, compress_gtiff=True, overwrite_products=False):
        self.products_df = products_df
        self.directory = directory
        self.compress_gtiff = compress_gtiff
        self.overwrite_products = overwrite_products

    def unzip_product(self, product, use_tile_path=False):
        if not product['product_path'].values[0].exists() or self.overwrite_products:
            file_name = product['title'].values[0]
            with zipfile.ZipFile((self.directory / 'zipfiles' / file_name).with_suffix('.zip'), "r") as zip_ref:
                if use_tile_path:
                    zip_ref.extractall(product['tile_path'].values[0])
                else:
                    zip_ref.extractall((product['product_path'].values[0]).parent)
            logging.debug('Product has been unzipped: ' + product['title'].values[0])
        else:
            logging.debug('Unzipped product already exists: ' + product['title'].values[0])

    def reproject_image(self, src, dst, crs='EPSG:32632'):
        src_tmp = None
        if src == dst:  # If you want to replace the src file with the output of this function
            src_tmp = Path(src.parent, 'temp' + src.suffix)  # Create path for temp file
            src.rename(src_tmp)  # Rename the file to the temp filename
            src = src_tmp  # Set the temp file to be the source file
        src_ds = gdal.Open(str(src))  # src is a Path object but gdal requires a string, therefore str(src)
        creation_options = ['NUM_THREADS=ALL_CPUS']
        config_options = ['GDAL_CACHEMAX=8192']
        warp_options = gdal.WarpOptions(dstSRS=crs, resampleAlg='cubic', srcNodata="0 0 0", multithread=True,
                                        creationOptions=creation_options)
        dst_ds = gdal.Warp(str(dst), src_ds, options=warp_options, config_options=config_options)
        dst_ds = None
        src_ds = None

        if src_tmp is not None:
            os.remove(src_tmp)

    @staticmethod
    def create_cog(img, factors=[2, 4, 8, 16, 32], compression='NONE'):
        # Define a new configuration, save the previous configuration,
        # and then apply the new one.
        new_config = {
            'GDAL_CACHEMAX': '8192',
            'COMPRESS_OVERVIEW': 'NONE'  # Compression will be made in the next step
        }
        prev_config = {
            key: gdal.GetConfigOption(key) for key in new_config.keys()}
        for key, val in new_config.items():
            gdal.SetConfigOption(key, val)

        InputImage = str(img)
        Image = gdal.Open(InputImage, 1)  # 0 = read-only, 1 = read-write.
        Image.BuildOverviews("NEAREST", factors)
        del Image

        # Restore previous configuration.
        for key, val in prev_config.items():
            gdal.SetConfigOption(key, val)


        # Run gdal_translate to properly tile and compress gtiff file after overviews have been added
        # Note: Should not be necessary, but the current version of gdal destroys tiles when adding overviews
        img_tmp = Path(img.parent, 'temp' + img.suffix)
        img.rename(img_tmp)

        ds = gdal.Open(str(img_tmp))  # img is a Path object but gdal requires a string, therefore str(img)
        # NOTE: ZSTD and WEBP compression is not working here
        # (https://lists.osgeo.org/pipermail/gdal-dev/2018-November/049289.html)
        if compression == 'NONE':
            creation_options = ['TILED=YES', 'COPY_SRC_OVERVIEWS=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512',
                                'NUM_THREADS=ALL_CPUS', 'BIGTIFF=IF_SAFER']
        elif compression == 'DEFLATE':
            creation_options = ['COMPRESS=DEFLATE', 'TILED=YES', 'COPY_SRC_OVERVIEWS=YES',
                                'BLOCKXSIZE=512', 'BLOCKYSIZE=512', 'NUM_THREADS=ALL_CPUS', 'BIGTIFF=IF_SAFER']
        elif compression == 'JPEG':
            creation_options = ['COMPRESS=JPEG', 'PHOTOMETRIC=YCBCR', 'JPEG_QUALITY=90', 'TILED=YES',
                                'COPY_SRC_OVERVIEWS=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512', 'NUM_THREADS=ALL_CPUS',
                                'BIGTIFF=IF_SAFER']
        config_options = ['GDAL_CACHEMAX=8192']
        translate_options = gdal.TranslateOptions(creationOptions=creation_options)
        ds = gdal.Translate(str(img), ds, options=translate_options, config_options=config_options)
        ds = None

        os.remove(img_tmp)

    def create_vrt(self, src_paths, vrt_path):
        vrt_options = gdal.BuildVRTOptions(addAlpha=False, hideNodata=False, allowProjectionDifference=False,
                                           resolution='highest')
        if not vrt_path.exists() or self.overwrite_products:
            logging.debug("Creating vrt file: " + str(vrt_path))
            if not vrt_path.parent.exists():  # Create directory if it does not exist
                os.makedirs(vrt_path.parent)
            vrt = gdal.BuildVRT(str(vrt_path), src_paths, options=vrt_options)
            vrt = None  # Required for saving the vrt (https://gis.stackexchange.com/a/314580)
        else:
            logging.debug("vrt file already exists: " + str(vrt_path))

    def vrt_to_geotiff(self, vrt_path, img_path):
        if not img_path.exists() or self.overwrite_products:
            ds = gdal.Open(str(vrt_path))
            creation_options = ['TILED=YES', 'COPY_SRC_OVERVIEWS=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512',
                                'NUM_THREADS=ALL_CPUS', 'BIGTIFF=IF_SAFER']
            config_options = ['GDAL_CACHEMAX=8192']
            translate_options = gdal.TranslateOptions(creationOptions=creation_options)
            ds = gdal.Translate(destName=str(img_path), srcDS=ds, options=translate_options,
                                config_options=config_options)
            ds = None


    @staticmethod
    def create_thumbnail(src, dst):
        translate_options = gdal.TranslateOptions(widthPct=3.25, heightPct=3.25, format='PNG')
        out_ds = gdal.Translate(str(dst), str(src), options=translate_options)
        out_ds = None  # Not sure if this is necessary
