import os
import shutil
import tenacity

from collections import OrderedDict
from sentinelsat import SentinelAPI
from pathlib import Path
from absl import flags, logging

FLAGS = flags.FLAGS


class Downloader(object):
    def __init__(self, username, password, satellite, order_id, directory=Path('/data/')):
        # The connection to ESA scihub
        self.api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')

        # Sentinel-5p currently has its own pre-operations hub
        self.api_s5p = SentinelAPI(user='s5pguest', password='s5pguest', api_url='https://s5phub.copernicus.eu/dhus')

        # Use the current datetime to name the download order
        self.order_id = order_id

        # Use ordered dict to store the metadata of the queries products
        self.products = OrderedDict()

        self.satellite = satellite

        self.directory = directory
        # if not self.directory.exists():  # Create directory if it does not exist
        #     os.makedirs(self.directory)

    def query(self, footprint, startdate, enddate):
        if self.satellite == 's1' or self.satellite == 'all':
            self.query_s1(footprint, startdate, enddate)
        if self.satellite == 's2' or self.satellite == 'all':
            self.query_s2(footprint, startdate, enddate)
        if self.satellite == 's3' or self.satellite == 'all':
            self.query_s3(footprint, startdate, enddate)
        if self.satellite == 's5p' or self.satellite == 'all':
            self.query_s5p(footprint, startdate, enddate)

    def query_s1(self, footprint, startdate, enddate):
        # Define producttypes (here it is Sentinel-1 GRDH products)
        producttypes = ['GRD']

        # Loop over producttypes and update the query dictionary
        # TODO: Fix this inefficient way of querying the relative orbits
        if FLAGS.s1_relative_orbit == [0]:
            for producttype in producttypes:
                queried_products = self.api.query(footprint,
                                                  date=(startdate, enddate),
                                                  platformname='Sentinel-1',
                                                  producttype=producttype,
                                                  sensoroperationalmode='IW')
                self.products.update(queried_products)

        else:
            for producttype in producttypes:
                for relative_orbit in FLAGS.s1_relative_orbit:
                    queried_products = self.api.query(footprint,
                                                      date=(startdate, enddate),
                                                      platformname='Sentinel-1',
                                                      producttype=producttype,
                                                      sensoroperationalmode='IW',
                                                      relativeorbitnumber=relative_orbit)
                    self.products.update(queried_products)

    def query_s2(self, footprint, startdate, enddate):
        # Load parameters from FLAGS
        max_cloudcoverage = FLAGS.s2_max_cloudcoverage

        # Define producttypes (here it is Sentinel-2 L2A products)
        producttypes = ['S2MSI2Ap', 'S2MSI2A']  # Producttype names differ depending on the year they were published

        # Loop over producttypes and update the query dictionary
        # TODO: Fix this inefficient way of querying the relative orbits
        if FLAGS.s2_relative_orbit == [0]:
            for producttype in producttypes:
                queried_products = self.api.query(footprint,
                                                  date=(startdate, enddate),
                                                  platformname='Sentinel-2',
                                                  producttype=producttype,
                                                  cloudcoverpercentage=(0, max_cloudcoverage))
                self.products.update(queried_products)

        else:
            for producttype in producttypes:
                for relative_orbit in FLAGS.s2_relative_orbit:
                    queried_products = self.api.query(footprint,
                                                      date=(startdate, enddate),
                                                      platformname='Sentinel-2',
                                                      relativeorbitnumber=relative_orbit,
                                                      producttype=producttype,
                                                      cloudcoverpercentage=(0, max_cloudcoverage))
                    self.products.update(queried_products)

    def query_s3(self, footprint, startdate, enddate):
        queried_products = self.api.query(footprint,
                                          date=(startdate, enddate),
                                          platformname='Sentinel-3',
                                          producttype='SL_2_LST___',
                                          productlevel='L2')

        self.products.update(queried_products)

    def query_s5p(self, footprint, startdate, enddate):
        kwargs = {}
        producttypedescriptions = ['Ozone', 'Sulphur Dioxide', 'Nitrogen Dioxide', 'Methane', 'Formaldehyde',
                                   'Carbon Monoxide', 'Aerosol Index', 'Aerosol Layer Height', 'Cloud']
        # producttypedescriptions = ['Ozone']

        # Loop over producttypes and update the query dictionary
        for producttypedescription in producttypedescriptions:
            queried_products = self.api_s5p.query(footprint,
                                                  date=(startdate, enddate),
                                                  platformname='Sentinel-5 Precursor',
                                                  processinglevel='L2',
                                                  producttypedescription=producttypedescription,
                                                  **kwargs)
            # Remove any 'Suomi-NPP VIIRS Clouds' products which are returned as 'Cloud' (they shouldn't have been)
            # https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-5p/products-algorithms
            if producttypedescription == 'Cloud':
                temp_queried_products = queried_products.copy()
                for key in queried_products.keys():
                    if queried_products[key]['producttypedescription'] != 'Cloud':
                        del temp_queried_products[key]
                queried_products = temp_queried_products
            self.products.update(queried_products)


    def print_num_and_size_of_products(self):
        logging.info('Number of products = ' + str(len(list(self.products))))
        logging.info('Total size [GB] = ' + str(self.api.get_products_size(self.products)))

    # https://sentinelsat.readthedocs.io/en/master/api.html#lta-products
    # TODO: Get LTA retrieval to work properly (install of newest sentinelsat version is in dockerfile)
    # Retry every 30 min (+10 second buffertime) to request LTA products.
    @tenacity.retry(stop=tenacity.stop_after_attempt(200), wait=tenacity.wait_fixed(1810))
    def download_zipfiles(self):
        zipfiles_directory = self.directory / 'zipfiles'
        if not zipfiles_directory.exists():  # Create directory if it does not exist
            os.makedirs(zipfiles_directory)

        # Get the products to be downloaded. The sample() funcitons permutes the dataframe, such that a new LTA product
        # is request at every retry. The optimal solution would have been to rearrange the dataframe by rotating the
        # index at every retry, but this is a quick and dirty way to achieve something similar.
        # (https://stackoverflow.com/a/34879805/12045808).
        products_df = self.queried_products_as_df().sample(frac=1)

        # NOTE: The code below is only useful while the Sentinel-5p has a different api than the others. After this has
        #       been fixed, the code should be reduced to the following single line:
        # Download all zipfiles (it automatically checks if zipfiles already exist)
        # self.api.download_all(self.products, directory_path=zipfiles_directory)  # Download all zipfiles
        # But for now, use the following code:
        non_s5p_products = products_df[products_df['platformname'] != 'Sentinel-5 Precursor']
        s5p_products = products_df[products_df['platformname'] == 'Sentinel-5 Precursor']

        if len(non_s5p_products):
            logging.info("Downloading Sentinel-1/2/3 products")
            self.api.download_all(non_s5p_products.to_dict(into=OrderedDict, orient='index'),
                                  directory_path=zipfiles_directory)
        else:
            logging.info("No Sentinel-1/2/3 products found in query")

        if len(s5p_products):
            logging.info("Downloading Sentinel-5p products")
            self.api_s5p.download_all(s5p_products.to_dict(into=OrderedDict, orient='index'),
                                      directory_path=zipfiles_directory)
        else:
            logging.info("No Sentinel-5p products found in query")

        # The Sentinel-5p data has wrongly been given the filetype .zip, but it should be .nc, so make a copy with
        # .nc extension. A copy is made instead of renaming so sentinelsat doesn't re-download the file every time
        # it is run.
        s5p_downloaded_files = zipfiles_directory.glob('S5P*.zip')
        logging.debug("Renaming downloaded Sentinel-5p files from .zip to .nc (due to bug in SentinelSat)")
        for file in s5p_downloaded_files:
            if not file.with_suffix('.nc').exists():
                shutil.copy(str(file), str(file.with_suffix('.nc')))

    def queried_products_as_geojson(self):
        return self.api.to_geojson(self.products)

    def queried_products_as_df(self):
        return self.api.to_dataframe(self.products)

    def save_queried_products(self):
        orders_directory = self.directory / 'orders'
        if not orders_directory.exists():
            os.makedirs(orders_directory)

        # Save the queried products to a geojson file (e.g. to be loaded into QGIS)
        geojson_path = (self.directory / 'orders' / self.order_id).with_suffix('.geojson')
        with geojson_path.open('w') as geojson_file:
            geojson_data = self.api.to_geojson(self.products)
            geojson_file.write(str(geojson_data))

        # Save the queried products as pandas df in a pkl file (preferred format when working in Python)
        df_path = (self.directory / 'orders' / self.order_id).with_suffix('.pkl')
        df = self.api.to_dataframe(self.products)
        df.to_pickle(df_path)

