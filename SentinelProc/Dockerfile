FROM osgeo/gdal:ubuntu-small-3.0.2 as builder

# Python dependencies that require compilation
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install -y python3-pip

# Required for pyproj to be installed (https://github.com/pyproj4/pyproj/issues/177#issuecomment-471292142)
RUN pip3 install -U pip

COPY requirements.txt /

RUN pip3 --no-cache-dir install -r /requirements.txt && \
    pip3 --no-cache-dir install rasterio==1.0.25 && \
    pip3 --no-cache-dir install scikit-image==0.15.0 && \
    pip3 --no-cache-dir install geopandas==0.5.1 && \
    pip3 --no-cache-dir install cmocean==2.0 && \
    pip3 --no-cache-dir install pyresample==1.12.3 && \
    pip3 --no-cache-dir install pykrige==1.4.1 && \
    pip3 --no-cache-dir install pyfftw==0.11.1 && \
    pip3 --no-cache-dir install arosics==0.9.3

# Install basemap
RUN apt-get install libgeos-dev && \
    pip3 --no-cache-dir install https://github.com/matplotlib/basemap/archive/master.zip

# Install the SNAP toolbox
# (https://github.com/schwankner/docker-esa-snap/blob/master/Dockerfile)
RUN wget http://step.esa.int/downloads/7.0/installers/esa-snap_sentinel_unix_7_0.sh && \
    chmod +x esa-snap_sentinel_unix_7_0.sh && \
    ./esa-snap_sentinel_unix_7_0.sh -q && \
    snap --nosplash --nogui --modules --update-all

# ------ Second stage
# Start from a clean image
FROM osgeo/gdal:ubuntu-small-3.0.2 as runner

# Install the previously-built libaries from the builder image
COPY --from=builder /usr/local /usr/local

# Arosics is annoying and want the following copies too
COPY --from=builder /usr/lib/python3 /usr/lib/python3
COPY --from=builder /usr/lib/python3.6 /usr/lib/python3.6
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/libgomp.so.1

# Create symlink for gpt (the graph processing tool in snap)
RUN ln -s /usr/local/snap/bin/gpt /usr/bin/gpt

# Set snap gpt max memory (not sure if this is necessary)
#RUN sed -i -e 's/-Xmx1G/-Xmx4G/g' /usr/local/snap/bin/gpt.vmoptions

# NOTE: THIS SHOULD BE REMOVED WHEN PYPI SENTINELSAT IS UPDATED!! IT USES A LOT OF SPACE!!
# Install master branch of sentinelsat (until they update pypi version)
RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y git && \
    git clone https://github.com/sentinelsat/sentinelsat.git /tmp/sentinelsat && \
    pip3 install /tmp/sentinelsat

# Create symlink to use 'python' instead of 'python3' to execute scripts
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create user such that output files will be owned by 1000:1000 (uid:gid).
# Note that only setting USER 1000:1000 in this dockerfile will results in execution errors with ESA SNAP gpt.
RUN useradd -ms /bin/bash sentinelproc-user && \
    chown -R sentinelproc-user /usr/local
USER sentinelproc-user:sentinelproc-user

# Copy the current directory contents into the container at /pythonapp
COPY . /pythonapp

# Set the working directory to /app
WORKDIR /pythonapp/app

# Run app.py when the container launches
ENTRYPOINT ["python", "app.py"]
