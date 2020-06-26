FROM jupyter/scipy-notebook:7a0c7325e470

ENV PYTHONDONTWRITEBYTECODE=true

# ------------------ INSTALL CONDA PACKAGES ------------------
# Ensure that we use MKL (get benchmark code here: http://markus-beuckelmann.de/blog/boosting-numpy-blas.html)
# Also see here: https://docs.anaconda.com/mkl-optimizations/
# There is a bug in their instructions however, and you need to add "blas=*=*mkl"
RUN conda install -yf mkl \
    && conda install -y numpy scipy scikit-learn numexpr "blas=*=*mkl"

# Install GDAL
RUN conda install --yes --freeze-installed -c conda-forge gdal==3.0.2 \
    && fix-permissions $CONDA_DIR \
    && fix-permissions /home/$NB_USER

# Clean up after installation
RUN conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete

# ------------------ INSTALL SNAP TOOLBOX ------------------
# (https://github.com/schwankner/docker-esa-snap/blob/master/Dockerfile)
# Install libgfotran3 due to following issue: https://forum.step.esa.int/t/error-in-sar-image-corregistration-error-nodeld-createstack-org-jblas-nativeblas-dgemm-cciiid-dii-diid-dii-v/12023/3
USER root
RUN wget http://step.esa.int/downloads/7.0/installers/esa-snap_sentinel_unix_7_0.sh && \
    chmod +x esa-snap_sentinel_unix_7_0.sh && \
    ./esa-snap_sentinel_unix_7_0.sh -q && \
    rm -f esa-snap_sentinel_unix_7_0.sh && \
    snap --nosplash --nogui --modules --update-all && \
    apt-get update && \
    apt-get install -y libgfortran3

# Clean apt-get cache (https://docs.docker.com/v17.09/engine/userguide/eng-image/dockerfile_best-practices/#run)
RUN rm -rf /var/lib/apt/lists/*

# Create symlink for gpt (the graph processing tool in snap)
RUN ln -s /opt/snap/bin/gpt /usr/bin/gpt
USER jovyan

# ------------------ INSTALL PYPI PACKAGES ------------------
# Install Python packages with pip
COPY requirements.txt /tmp/
RUN pip --no-cache-dir install -r /tmp/requirements.txt

# ------------------ CONFIGURE THE LAST THINGS ------------------
# Copy the code and set the working
COPY . /home/jovyan/work
WORKDIR /home/jovyan/work