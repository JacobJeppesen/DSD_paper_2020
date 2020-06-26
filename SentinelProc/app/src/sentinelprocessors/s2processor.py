import os
import rasterio

import numpy as np

from arosics import COREG, DESHIFTER
from absl import logging, flags
from pathlib import Path

from src.sentinelprocessors.base import BaseProcessor

FLAGS = flags.FLAGS


class S2Processor(BaseProcessor):
    def __init__(self, products_df, directory, compress_gtiff=True, overwrite_products=False, delete_jp2_files=False):
        super().__init__(products_df, directory, compress_gtiff=compress_gtiff, overwrite_products=overwrite_products)
        self.calc_ndvi = FLAGS.s2_ndvi
        self.compress_gtiff = FLAGS.compress_gtiff
        self.delete_jp2_files = FLAGS.delete_jp2_files
        self.overwrite_products = FLAGS.overwrite

    def process_tiles(self, index):
        # Extract the row from the dataframe with the product process (use copy to avoid "copy of a slice warning")
        # https://stackoverflow.com/questions/31468176/setting-values-on-a-copy-of-a-slice-from-a-dataframe
        product = self.products_df.iloc[[index]].copy()
        self.unzip_product(product, use_tile_path=True)
        product = self.__concat_resolutions_paths(product)  # Must be done after unzipping (damnit ESA!)
        self.__translate_jp2_to_gtiff(product)
        self.__calculate_ndvi__(product)
        self.__calculate_gndvi__(product)
        # self.__calculate_gndvi_gdal__(product)

    @staticmethod
    def __concat_resolutions_paths(product):
        granule_path = product['product_path'].values[0] / 'GRANULE'
        resolutions_path = list(granule_path.glob('*/'))[0] / 'IMG_DATA'
        product['res_10m_path'] = resolutions_path / 'R10m/'
        product['res_20m_path'] = resolutions_path / 'R20m/'
        product['res_60m_path'] = resolutions_path / 'R60m/'
        return product

    def __translate_jp2_to_gtiff(self, product):
        # Translate all files to geotiff
        resolution_paths = [product['res_10m_path'].values[0],
                            product['res_20m_path'].values[0],
                            product['res_60m_path'].values[0]]

        for resolution_path in resolution_paths:
            jp2_files = list(resolution_path.glob('*.jp2'))
            jp2_files.sort()
            for jp2_file in jp2_files:
                gtiff_file = jp2_file.with_suffix('.tiff')
                if not gtiff_file.exists() or self.overwrite_products:
                    self.reproject_image(jp2_file, gtiff_file)
                    if self.compress_gtiff:
                        if 'TCI' in gtiff_file.stem:
                            # self.create_cog(gtiff_file, compression='JPEG')
                            # logging.debug('Jpeg2000 file has been converted to Gtiff with JPEG compression: ' +
                            #               str(gtiff_file))
                            self.create_cog(gtiff_file, compression='DEFLATE')
                            logging.debug('Jpeg2000 file has been converted to Gtiff with DEFLATE compression: ' +
                                          str(gtiff_file))
                        else:
                            self.create_cog(gtiff_file, compression='DEFLATE')
                            logging.debug('Jpeg2000 file has been converted to Gtiff with DEFLATE compression: ' +
                                          str(gtiff_file))
                    else:
                        self.create_cog(gtiff_file, compression='NONE')
                        logging.debug('Jpeg2000 file has been converted to Gtiff with no compression: ' +
                                      str(gtiff_file))

                    # Delete jp2 files
                    if self.delete_jp2_files:
                        os.remove(jp2_file)
                        logging.info('Jpeg2000 file has been deleted: ' + str(jp2_file))

                    logging.info("Sentinel-2 file has been translated to GeoTiff: " + str(gtiff_file))
                else:
                    logging.info("Sentinel-2 file already exists as GeoTiff: " + str(gtiff_file))

    def __calculate_ndvi__(self, product, use_acorvi=True):
        # ACORVI: Regarding use of visible bands for vegetation indicies with atmospherically corrected data
        # https://labo.obs-mip.fr/multitemp/using-ndvi-with-atmospherically-corrected-data/
        res_10m_path = product['res_10m_path'].values[0]
        red_band_path = list(res_10m_path.glob('*B04_10m.tiff'))[0]
        nir_band_path = list(res_10m_path.glob('*B08_10m.tiff'))[0]
        ndvi_path = Path(red_band_path.parent, str(red_band_path.stem)[:-7] + 'NDVI_10m' + red_band_path.suffix)

        if not ndvi_path.exists() or self.overwrite_products:
            with rasterio.open(red_band_path) as red:
                with rasterio.open(nir_band_path) as nir:
                    profile = red.profile
                    # Is compression needed here? We make a cog afterwards anyways.
                    # if self.compress_gtiff:
                    #     profile.update(dtype=rasterio.float32, compress='DEFLATE', predictor=3, zlevel=9)
                    # else:
                    #     profile.update(dtype=rasterio.float32)
                    profile.update(dtype=rasterio.float32)
                    red_data = red.read().astype(rasterio.float32)
                    nir_data = nir.read().astype(rasterio.float32)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        if use_acorvi:
                            # https://labo.obs-mip.fr/multitemp/using-ndvi-with-atmospherically-corrected-data/
                            # ndvi_data = (nir_data - red_data + 0.05) / (nir_data + red_data + 0.05)
                            ndvi_data = (nir_data - red_data) / (nir_data + red_data)
                        else:
                            ndvi_data = (nir_data - red_data) / (nir_data + red_data)
                        with rasterio.open(ndvi_path, 'w', **profile) as dst:
                            # Round the NDVI data to 2 decimals (reduces the size after compression) and save the data
                            dst.write(np.round(ndvi_data, decimals=2))
                    self.create_cog(ndvi_path, compression='DEFLATE')
            logging.info("Sentinel-2 product has had NDVI calculated: " + product['title'].values[0])
        else:
            logging.info("Sentinel-2 product already had NDVI calculated: " + product['title'].values[0])

    def __calculate_gndvi__(self, product, use_acorvi=True):
        # ACORVI: Regarding use of visible bands for vegetation indicies with atmospherically corrected data
        # https://labo.obs-mip.fr/multitemp/using-ndvi-with-atmospherically-corrected-data/
        res_10m_path = product['res_10m_path'].values[0]
        green_band_path = list(res_10m_path.glob('*B03_10m.tiff'))[0]
        nir_band_path = list(res_10m_path.glob('*B08_10m.tiff'))[0]
        gndvi_path = Path(green_band_path.parent, str(green_band_path.stem)[:-7] + 'GNDVI_10m' + green_band_path.suffix)

        if not gndvi_path.exists() or self.overwrite_products:
            with rasterio.open(green_band_path) as green:
                with rasterio.open(nir_band_path) as nir:
                    profile = green.profile
                    # Is compression needed here? We make a cog afterwards anyways.
                    # if self.compress_gtiff:
                    #     profile.update(dtype=rasterio.float32, compress='DEFLATE', predictor=3, zlevel=9)
                    # else:
                    #     profile.update(dtype=rasterio.float32)
                    profile.update(dtype=rasterio.float32)
                    green_data = green.read().astype(rasterio.float32)
                    nir_data = nir.read().astype(rasterio.float32)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        if use_acorvi:
                            # https://labo.obs-mip.fr/multitemp/using-ndvi-with-atmospherically-corrected-data/
                            # gndvi_data = (nir_data - green_data + 0.05) / (nir_data + green_data + 0.05)
                            gndvi_data = (nir_data - green_data) / (nir_data + green_data)
                        else:
                            gndvi_data = (nir_data - green_data) / (nir_data + green_data)
                        with rasterio.open(gndvi_path, 'w', **profile) as dst:
                            # Round the NDVI data to 2 decimals (reduces the size after compression) and save the data
                            dst.write(np.round(gndvi_data, decimals=2))
                    self.create_cog(gndvi_path, compression='DEFLATE')
            logging.info("Sentinel-2 product has had GNDVI calculated: " + product['title'].values[0])
        else:
            logging.info("Sentinel-2 product already had GNDVI calculated: " + product['title'].values[0])

    def __calculate_gndvi_gdal__(self, product):
        res_10m_path = product['res_10m_path'].values[0]
        green_band_path = list(res_10m_path.glob('*B03_10m.tiff'))[0]
        nir_band_path = list(res_10m_path.glob('*B08_10m.tiff'))[0]
        gndvi_path = Path(green_band_path.parent, str(green_band_path.stem)[:-7] + 'GNDVI_10m' + green_band_path.suffix)

        if not gndvi_path.exists() or self.overwrite_products:
            # Arguements.
            input_file_path_A = str(nir_band_path)
            input_file_path_B = str(green_band_path)
            output_file_path = str(gndvi_path)
            calc_expr = '"1.0*(1.0*A - 1.0*B) / 1.0*(1.0*A + 1.0*B)"'
            typeof = '"Float32"'

            # Generate string of process.
            gdal_calc_str = 'gdal_calc.py -A {0} -B {1} --outfile={2} --calc={3} --type={4} --overwrite'
            gdal_calc_process = gdal_calc_str.format(input_file_path_A, input_file_path_B,
                                                     output_file_path, calc_expr, typeof)


            # Call process.
            os.system(gdal_calc_process)
            # self.create_cog(gndvi_path, compression='DEFLATE')
            logging.info("Sentinel-2 product has had GNDVI calculated: " + product['title'].values[0])
        else:
            logging.info("Sentinel-2 product already had GNDVI calculated: " + product['title'].values[0])

    def create_vrt_files_and_coregister(self, modes=['TCI', 'NDVI', 'GNDVI']):
        product_titles = self.products_df['title'].values
        product_dates = [title[11:19] for title in product_titles]
        product_dates = np.unique(product_dates)
        logging.info("Creating vrt file(s)")
        for product_date in product_dates:
            self.create_vrt_and_coregister_modes(modes, product_date)

    def create_vrt_and_coregister_modes(self, modes, product_date):
        # TODO: This function, along with coregister, should be written more elegantly
        coreg_info = None  # Co-registration object (http://danschef.gitext.gfz-potsdam.de/arosics/doc/usage/global_coreg.html)
        for mode in modes:
            relative_orbit, tile_paths = self.__get_tile_paths(product_date, mode=mode)
            vrt_name = 'S2_L2A_' + product_date + '_' + relative_orbit + '_' + mode + '.vrt'
            vrt_path = Path(self.directory / 'output_data' / 's2' / 'combined' / vrt_name)
            self.create_vrt(src_paths=tile_paths, vrt_path=vrt_path)
            if mode == 'TCI':
                self.create_thumbnail(src=vrt_path, dst=vrt_path.with_suffix('.png'))

            # Co-register and create geotiff
            dst_path = Path(str(vrt_path)[:-4] + '_coreg.tiff')
            if coreg_info is None:  # Find the spatial shift (ie. for the first mode)
                coreg_info = self.coregister(src_path=vrt_path, dst_path=dst_path)
            elif coreg_info is not 'coreg_error':  # Apply to other modes
                try:
                    DESHIFTER(im2shift=str(vrt_path), path_out=str(dst_path), fmt_out='GTIFF',
                              coreg_results=coreg_info).correct_shifts()
                    self.create_cog(dst_path, compression='DEFLATE')
                except:
                    logging.error("Coregistration error with file: " + str(dst_path))
            else:  # Log an error if the coregistration has failed
                logging.error("Coregistration error with file: " + str(dst_path))

    def coregister(self, src_path, dst_path):
        try:
            logging.debug("Coregistering image: " + str(dst_path))
            if not dst_path.exists() or self.overwrite_products:
                compress_option = 'COMPRESS=NONE'  # This is only used for the intermediate file and does not need compression

                # Remember to set nodata value to 0 when saving the ortofoto
                reference_im_path = Path('data/reference')
                im_reference = str(reference_im_path / 'ortofoto_reference_epsg32632.tif')
                im_target = str(src_path)
                kwargs = {
                    'mask_baddata_ref': str(reference_im_path / 'ortofoto_reference_epsg32632_nodata_mask.tif'),
                    'path_out': str(dst_path),
                    'fmt_out': 'GTIFF',
                    'out_crea_options': [compress_option],
                    'ws': (4096, 4096),
                    'q': False,
                    'v': False
                }

                CR = COREG(im_reference, im_target, **kwargs)
                CR.calculate_spatial_shifts()
                logging.info("Shift reliability is " + str(round(CR.shift_reliability, 2)) + "% for " + str(src_path.name))

                CR.correct_shifts()
                self.create_cog(dst_path, compression='DEFLATE')
                coreg_info = CR.coreg_info

                return coreg_info
            else:
                logging.debug("Coregistered image already exist: " + str(dst_path))
        except:
            logging.error("Coregistration error with file: " + str(src_path))
            CR = 'coreg_error'
            return CR

    def __get_tile_paths(self, product_date, mode='TCI'):
        tile_paths = []  # Placeholder for the paths to all TCI products
        for index in range(len(self.products_df.index)):
            product = self.products_df.iloc[[index]].copy()
            product = self.__concat_resolutions_paths(product)
            res_10m_path = product['res_10m_path'].values[0]
            tile_path = list(res_10m_path.glob('*_' + mode + '_10m.tiff'))[0]
            # file_names = os.listdir(product_paths['res_10m'])
            # tci_path = product_paths['res_10m'] + [f for f in file_names if 'TCI_10m.tiff' in f][0]
            product_date_current = product['title'].values[0][11:19]
            if product_date_current == product_date:
                tile_paths.append(str(tile_path))
                relative_orbit = product['title'].values[0][33:37]
        return relative_orbit, tile_paths

