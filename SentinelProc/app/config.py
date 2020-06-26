from absl import flags

FLAGS = flags.FLAGS


def define_flags():
    #####################
    # General app flags #
    #####################
    flags.DEFINE_string('username', 'username', 'Username for scihub.copernicus.eu')
    flags.DEFINE_string('password', 'password', 'Username for scihub.copernicus.eu')
    flags.DEFINE_string('data_directory', '/data/', 'Data directory used for storing downloaded and processed data')
    flags.DEFINE_bool('download', True, 'Download the files')
    flags.DEFINE_bool('process_tiles', True, 'Process the downloaded files')
    flags.DEFINE_bool('compress', True, 'Compress pre- and post-processed geotiff files')
    flags.DEFINE_string('logging_verbosity', 'info', 'Logging verbosity (debug|info|warning|error|fatal).')

    ###############
    # Query flags #
    ###############
    flags.DEFINE_string('order_id', 'Empty', 'Order id for postprocessing of already downloaded file')
    flags.DEFINE_string('geojson', 'denmark_without_bornholm', 'Name of the geojson file defining the area to query')
    flags.DEFINE_string('startdate', '20190803', 'The sensing start date')
    flags.DEFINE_string('enddate', '20190808', 'The sensing end date')
    flags.DEFINE_string('satellite', 'all', 'The Sentinel satellite(s) to get data from (s1/s2/s3/s5p/all)')
    # Sentinel-1
    flags.DEFINE_multi_integer('s1_relative_orbit', [44], 'Relative orbit number (0 => all relative orbits)')
    # Sentinel-2
    flags.DEFINE_multi_integer('s2_relative_orbit', [8, 108], 'Relative orbit number (0 => all relative orbits)')
    flags.DEFINE_integer('s2_max_cloudcoverage', 100, 'The maximum allowed cloud coverage')
    # Sentinel-3
    # Sentinel-5p

    ###################
    # Processor flags #
    ###################
    # General
    flags.DEFINE_bool('overwrite', True, 'Overwrite existing products')
    flags.DEFINE_bool('compress_gtiff', True, 'Compress GTiff files')
    # Sentinel-1
    flags.DEFINE_integer('s1_num_proc', 2,
                         'Number of parallel processes for Sentinel-1 processing (approx. 20 GB RAM per process)')
    flags.DEFINE_bool('s1_del_intermediate', False, 'Delete the intermediate Sentinel-1 processing data')
    flags.DEFINE_string('s1_output_crs', 'EPSG:32632', 'Coordinate reference system for the output combined geotiff')
    # Sentinel-2
    flags.DEFINE_integer('s2_num_proc', 6,
                         'Number of parallel processes for Sentinel-1 processing (approx. 8 GB RAM per process)')
    flags.DEFINE_bool('delete_jp2_files', True, 'Delete jp2 files after they have been converted to GTiff')
    flags.DEFINE_bool('s2_ndvi', True, 'Calculate NDVI index of Sentinel-2 data')
    # Sentinel-3
    # Sentinel-5p

