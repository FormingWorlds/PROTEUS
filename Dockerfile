# PROTEUS Docker Image - Pre-built Environment with Compiled Physics Modules
# This image contains a ready-to-run PROTEUS environment with all compiled physics modules.
# It is built nightly and used by CI/CD for fast testing.

FROM python:3.12-slim-bookworm

# Metadata
LABEL maintainer="tim.lichtenberg@rug.nl"
LABEL description="PROTEUS ecosystem with pre-compiled physics modules"
LABEL org.opencontainers.image.source="https://github.com/FormingWorlds/PROTEUS"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FWL_DATA=/opt/proteus/fwl_data \
    RAD_DIR=/opt/proteus/socrates \
    AGNI_DIR=/opt/proteus/AGNI \
    PETSC_DIR=/opt/proteus/petsc \
    PETSC_ARCH=arch-linux-c-opt \
    PROTEUS_DIR=/opt/proteus \
    JULIA_NUM_THREADS=1

# Install system dependencies (matching docs/installation.md)
# - gfortran: Fortran compiler for SOCRATES and SPIDER
# - make, cmake: Build tools
# - git: Version control
# - libnetcdff-dev, netcdf-bin: NetCDF libraries for Fortran
# - libssl-dev: SSL support
# - curl, wget: Download tools
# - unzip: Archive extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    gcc \
    g++ \
    make \
    cmake \
    git \
    libnetcdff-dev \
    netcdf-bin \
    libssl-dev \
    curl \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Julia (matching docs/installation.md)
RUN curl -fsSL https://install.julialang.org | sh -s -- -y && \
    ln -s /root/.juliaup/bin/julia /usr/local/bin/julia

# Create working directory
WORKDIR /opt/proteus

# Copy source code for compilation
# This ensures the image contains the exact source code state
COPY . /opt/proteus/

# Install Python dependencies from pyproject.toml
# Developer install for editable mode to allow code overlay in CI
RUN pip install --upgrade pip && \
    pip install -e ".[develop]"

# Build SOCRATES (Radiative transfer code)
# This is the most time-consuming compilation step
RUN cd /opt/proteus && \
    ./tools/get_socrates.sh && \
    echo "export RAD_DIR=/opt/proteus/socrates" >> /root/.bashrc

# Build PETSc (Numerical computing library)
RUN cd /opt/proteus && \
    ./tools/get_petsc.sh && \
    echo "export PETSC_DIR=/opt/proteus/petsc" >> /root/.bashrc && \
    echo "export PETSC_ARCH=arch-linux-c-opt" >> /root/.bashrc

# Build SPIDER (Interior evolution model)
RUN cd /opt/proteus && \
    ./tools/get_spider.sh && \
    chmod +x SPIDER/spider

# Build AGNI (Radiative-convective atmosphere model)
# Clone AGNI if not present (submodule)
RUN if [ ! -d "/opt/proteus/AGNI" ]; then \
        git clone https://github.com/nichollsh/AGNI.git /opt/proteus/AGNI; \
    fi && \
    cd /opt/proteus/AGNI && \
    bash src/get_agni.sh 0

# Install submodules as editable packages (developer workflow)
RUN if [ -d "/opt/proteus/MORS" ]; then pip install -e MORS/.; fi && \
    if [ -d "/opt/proteus/aragog" ]; then pip install -e aragog/.; fi && \
    if [ -d "/opt/proteus/JANUS" ]; then pip install -e JANUS/.; fi && \
    if [ -d "/opt/proteus/CALLIOPE" ]; then pip install -e CALLIOPE/.; fi && \
    if [ -d "/opt/proteus/ZEPHYRUS" ]; then pip install -e ZEPHYRUS/.; fi

# Create FWL_DATA directory for test data
RUN mkdir -p $FWL_DATA

# Clean up to reduce image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    find /opt/proteus -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/proteus -type f -name "*.pyc" -delete && \
    find /opt/proteus -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Set working directory for test execution
WORKDIR /opt/proteus

# Default command: show environment info
CMD ["bash", "-c", "echo 'PROTEUS Docker Image Ready' && python --version && julia --version"]
