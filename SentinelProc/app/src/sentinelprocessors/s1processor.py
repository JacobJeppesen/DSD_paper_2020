import os
import subprocess
from pathlib import Path

import numpy as np
import rasterio
import shutil
from absl import flags, logging
from matplotlib import pyplot as plt

from src.sentinelprocessors.base import BaseProcessor

FLAGS = flags.FLAGS


class S1Processor(BaseProcessor):
    def __init__(self, products_df, directory, compress_gtiff=True, overwrite_products=False):
        super().__init__(products_df, directory, compress_gtiff=compress_gtiff, overwrite_products=overwrite_products)
        self.compress_gtiff = FLAGS.compress_gtiff
        self.del_intermediate = FLAGS.s1_del_intermediate
        self.directory = directory
        self.overwrite_products = FLAGS.overwrite
        self.output_crs = FLAGS.s1_output_crs

    def process(self, index):
        # Extract the row from the dataframe with the product process (use copy to avoid "copy of a slice warning")
        # https://stackoverflow.com/questions/31468176/setting-values-on-a-copy-of-a-slice-from-a-dataframe
        product = self.products_df.iloc[[index]].copy()
        try:
            logging.info("Processing product: " + str(product['title'].values[0]))
            self.unzip_product(product)
            product_path = product['product_path'].values[0]

            # NOTE: The following function should not be necessary - I think it is in the product dataframe already
            manifest_path = product_path / 'manifest.safe'
            rel_orbit, pass_mode = self.get_rel_orbit_and_pass_mode(manifest_path)

            # Run preprocessing graph
            preprocessed_product_path = self.preprocess(product_path, rel_orbit, pass_mode)

            # Create geotiffs
            vh_path, vv_path, vv_vh_path = self.to_geotiff_bands(preprocessed_product_path)
            self.to_geotiff_rgb(vh_path, vv_path, vv_vh_path, product_path, dtype=rasterio.int16)

            # Delete intermediate data
            if self.del_intermediate:
                shutil.rmtree(product_path / 'preprocessed')

            logging.info("Finished processing product: " + str(product['title'].values[0]))
        except:
            logging.error("An error occured during proccesing of: " + str(product['title'].values[0]))

    def preprocess(self, product_path, rel_orbit, pass_mode):
        graph_path = Path('data') / 'graphs' / 'preprocessGraph.xml'
        specklefilter='None' # Use 'None' or 'Refined Lee' ('Refined Lee' might need "" or something around it to be parsed properly)
        dst_name = str(product_path.stem) + '_' + pass_mode + '_' + rel_orbit
        dst_path = (product_path / 'preprocessed' / dst_name).with_suffix('.dim')
        cmd = 'gpt {} -Pinput={} -Pspecklefilter={} -Poutput={}'.format(str(graph_path),
                                                                        str(product_path),
                                                                        str(specklefilter),
                                                                        str(dst_path))
        if not dst_path.exists() or self.overwrite_products:
            self.exec_graph(cmd)
            logging.debug('Sentinel-1 product has been preprocessed: ' + str(product_path))
        else:
            logging.debug('Sentinel-1 product has already been preprocessed: ' + str(product_path))

        return dst_path

    def to_geotiff_bands(self, preprocessed_product_path):
        graph_path = Path('data') / 'graphs' / 'geotiffFloat32Graph.xml'
        output_vh = Path(str(preprocessed_product_path.with_suffix('')) + '_VH.tif')
        output_vv = Path(str(preprocessed_product_path.with_suffix('')) + '_VV.tif')
        output_vv_vh = Path(str(preprocessed_product_path.with_suffix('')) + '_VV-VH.tif')

        cmd = 'gpt {} -Pinput={} -Poutput_vh={} -Poutput_vv={} -Poutput_vv-vh={}'.format(str(graph_path),
                                                                                         str(preprocessed_product_path),
                                                                                         str(output_vh),
                                                                                         str(output_vv),
                                                                                         str(output_vv_vh))

        if not output_vh.exists() or self.overwrite_products:
            self.exec_graph(cmd)
            logging.debug('Sentinel-1 product has had individual geotiff bands created: ' +
                          str(preprocessed_product_path))
        else:
            logging.debug('Sentinel-1 product already had individual geotiff bands created: ' +
                          str(preprocessed_product_path))

        return [output_vh, output_vv, output_vv_vh]

    def to_geotiff_rgb(self, vh_path, vv_path, vv_vh_path, product_path, dtype=rasterio.float32):
        # From https://gis.stackexchange.com/a/223920
        file_list = [vh_path, vv_path, vv_vh_path]
        dst_name = str(vh_path.stem)[:-2] + 'RGB.tif'
        dst_path = product_path / 'processed' / dst_name

        if not dst_path.exists() or self.overwrite_products:
            if not dst_path.parent.exists():  # Create directory if it does not exist
                os.makedirs(dst_path.parent)

            # Read metadata of first file
            with rasterio.open(file_list[0]) as src0:
                profile = src0.profile

            # Update meta to reflect the number of layers
            profile.update(
                count=len(file_list),
                nodata=-32768,
                dtype=dtype)

            # Read each layer and write it to stack
            with rasterio.open(dst_path, 'w', **profile) as dst:
                for i, layer in enumerate(file_list, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(i, src1.read(1).astype(dtype))

            # Create cloud optimized geotiff
            self.create_cog(dst_path, compression='DEFLATE')

            # Create thumbnail of the RGB image
            self.create_sentinel1_thumbnail(dst_path)

            logging.debug('Sentinel-1 product has had RGB geotiff created: ' + str(dst_path))
        else:
            logging.debug('Sentinel-1 product already had RGB geotiff created: ' + str(dst_path))

    def create_vrt_and_cog(self, product_date_abs_orbit):
        logging.info("Creating Sentinel-1 vrt and geotiff file(s) for date and abs orbit: " + product_date_abs_orbit)
        rgb_paths = []
        for index in range(len(self.products_df.index)):
            product = self.products_df.iloc[[index]].copy()
            rgb_path = list((product['product_path'].values[0] / 'processed').glob('*RGB.tif*'))[0]
            product_date = product['title'].values[0][17:25]
            product_abs_orbit = product['title'].values[0][48:55]
            if product_date_abs_orbit == (product_date + product_abs_orbit):
                rgb_paths.append(str(rgb_path))

                # The values below should also be in the products dataframe.
                satellite_name = product['title'].values[0][0:3]
                pass_mode = str(rgb_path.stem)[-11:-8]
                rel_orbit = str(rgb_path.stem)[-7:-4]

        vrt_name = satellite_name + '_' + product_date_abs_orbit + '_' + pass_mode + '_' + rel_orbit + '_RGB.vrt'
        vrt_path = Path(self.directory / 'output_data' / 's1' / 'combined' / vrt_name)
        self.create_vrt(src_paths=rgb_paths, vrt_path=vrt_path)

        img_path = vrt_path.with_suffix('.tif')
        # File check has to be done here or COG will be made every time (there might be better way to implement this)
        if not img_path.exists() or self.overwrite_products:
            self.vrt_to_geotiff(vrt_path, img_path)
            # TODO: Get reprojection to work properly
            # self.reproject_image(src=img_path, dst=img_path, crs=self.output_crs)
            self.create_cog(img_path, compression='DEFLATE')
            self.create_sentinel1_thumbnail(img_path)

    def create_parent_folders(self):
        product_paths = self.products_df['product_path'].values  # NumPy array of paths
        for product_path in product_paths:
            if not product_path.parent.exists():
                os.makedirs(product_path.parent)

    def create_sentinel1_thumbnail(self, img_path):
        thumbnail_path = img_path.with_suffix('.png')
        with rasterio.open(img_path) as src:
            # List of overviews from biggest to smallest
            oviews = src.overviews(1)

            # Retrieve the thumbnail
            oview = oviews[-1]
            # NOTE this is using a 'decimated read' (http://rasterio.readthedocs.io/en/latest/topics/resampling.html)
            out_shape = (int(src.height // oview), int(src.width // oview))
            thumbnail = src.read(out_shape=out_shape).astype(rasterio.float32)

            # Convert 0 values to NaNs
            thumbnail[thumbnail == -32768] = np.nan

            # Move channels to last axis and normalize with the colours manually found (ie. vh=+30, vv=+20, vv-vh=+0)
            thumbnail = np.rollaxis(thumbnail, 0, 3)
            thumbnail[:, :, 0] = np.clip((thumbnail[:, :, 0]+30) / 25, 0, 1)
            thumbnail[:, :, 1] = np.clip((thumbnail[:, :, 1]+20) / 25, 0, 1)
            thumbnail[:, :, 2] = np.clip((thumbnail[:, :, 2]) / 25, 0, 1)

        plt.imsave(thumbnail_path, thumbnail)

    @staticmethod
    def exec_graph(cmd):
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout.readlines():
            logging.debug(line.strip().decode())
        retval = p.wait()

    @staticmethod
    def get_rel_orbit_and_pass_mode(manifest_path):
        with open(manifest_path) as fh:
            for line in fh:
                if ('relativeOrbitNumber' in line and 'start' in line):
                    # Find the text between the tags in the line (it is from an XML document)
                    index1 = [i for i, char in enumerate(line) if char == '>'][0]  # Index of first '>'
                    index2 = [i for i, char in enumerate(line) if char == '<'][1]  # Index of second '<'
                    rel_orbit = line[index1+1:index2]
                if 's1:pass' in line:
                    # Find the text between the tags in the line (it is from an XML document)
                    index1 = [i for i, char in enumerate(line) if char == '>'][0]  # Index of first '>'
                    index2 = [i for i, char in enumerate(line) if char == '<'][1]  # Index of second '<'
                    pass_mode = line[index1+1:index2]

        # Ensure that rel_orbit is a 3 character string
        rel_orbit = format(int(rel_orbit), '03d')

        # Abbreviate pass_mode ('ASC'='ASCENDING', 'DSC'='DESCENDING')
        pass_mode = 'ASC' if pass_mode == 'ASCENDING' else 'DSC'

        return rel_orbit, pass_mode

