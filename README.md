# Crop Type Classification based on Machine Learning with Multitemporal Sentinel-1 Data
This repository contains the code for the paper titled "Crop Type Classification based on Machine Learning with Multitemporal Sentinel-1 Data", presented at the Digital System Design (DSD) 2020 conference.

## Running the code
The easiest way to run the code in the notebooks is through [Docker](https://www.docker.com/). Start by installing Docker, and then run the command below to launch a docker container with all notebooks, the processed data, and all packages installed. Jupyter Notebook can then be accessed by launching your browser and navigating to the url [localhost:8888](http://localhost:8888).
```
docker run --rm -p 8888:8888 -e GRANT_SUDO=yes -u root jacobjeppesen/dsd-paper-2020 start.sh jupyter notebook --NotebookApp.token=''
```

Alternatively, you can install the Python packages yourself from the requirements.txt file. It is recommended to do this in a virtual environment, using software such as [Anaconda](https://www.anaconda.com/). Then download the processed dataset from [https://aarhusuniversitet-my.sharepoint.com/:u:/g/personal/au280553_uni_au_dk/ETpL-u7HOCVOsTaGOg6iRX4BZc2-iEXbsVlMo0pzS78f9w?e=GiQS9t](https://aarhusuniversitet-my.sharepoint.com/:u:/g/personal/au280553_uni_au_dk/ETpL-u7HOCVOsTaGOg6iRX4BZc2-iEXbsVlMo0pzS78f9w?e=GiQS9t), put it under _data/processed_, and you should be good to go.

## Downloading new data
The processed data provided above (ie., the xarray dataset) contains all the data needed to run all experiments in the paper. We have not put the entire pre-processed dataset online, as it contains 294 GB of Sentinel-1 raster files. However, we have included the code used to download and pre-process the Sentinel-1 data in the SentinelProc folder. The SentinelProc project has been renamed to [satproc](https://github.com/JacobJeppesen/satproc), and is currently under development. If you want to use it now, however, the easiest way is to run the command below to use the designated Docker container. But note that you are likely to encounter errors, as the code is still at a prototype level.
```
docker pull jacobjeppesen/sentinelproc
docker run --rm -it -v $PWD:/data/ jacobjeppesen/sentinelproc \
        --username your_esa_scihub_username \
        --password your_esa_scihub_password \
        --data_directory /data/sentinelproc \
        --geojson denmark_without_bornholm \
        --satellite s1 \
        --startdate 20180703 \
        --enddate 20191101 \
        --s1_num_proc 4 \
        --s1_relative_orbit 139 \
        --s1_del_intermediate True \
        --s1_output_crs EPSG:32632 \
        --s2_num_proc 6 \
        --s2_relative_orbit 0
```