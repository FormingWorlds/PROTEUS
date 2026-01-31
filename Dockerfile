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
    PROTEUS_DIR=/opt/proteus \
    JULIA_NUM_THREADS=1 \
    JULIA_DEPOT_PATH=/opt/julia_depot

# Install system dependencies (matching docs/installation.md)
# - gfortran: Fortran compiler for SOCRATES and SPIDER
# - make, cmake: Build tools
# - git: Version control
# - libnetcdff-dev, netcdf-bin: NetCDF libraries for Fortran
# - libssl-dev: SSL support
# - curl, wget: Download tools
# - unzip: Archive extraction
# - rsync: File synchronization for CI code overlay
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
    rsync \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Julia 1.11 (required by AGNI - must match Project.toml compat)
# CRITICAL: AGNI requires Julia ~1.11 (not 1.12+) - version mismatch causes test failures
# Using direct download instead of juliaup to avoid broken symlinks
# Add to PATH instead of symlinking to preserve library paths
RUN JULIA_VERSION=1.11.2 && \
    JULIA_MINOR=1.11 && \
    curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_MINOR}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" -o julia.tar.gz && \
    tar -xzf julia.tar.gz -C /opt && \
    rm julia.tar.gz && \
    /opt/julia-${JULIA_VERSION}/bin/julia --version

# Add Julia to PATH
ENV PATH="/opt/julia-1.11.2/bin:${PATH}"

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

# Clone SPIDER for reference (not built - tests don't use it)
# Skipping PETSc download/build and SPIDER compilation to speed up image creation
RUN cd /opt/proteus && \
    mkdir -p SPIDER && \
    echo "SPIDER directory created for compatibility" > SPIDER/README.txt

# Configure git to use HTTPS for all GitHub operations (avoid SSH dependency)
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

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

# Create FWL_DATA and Julia depot directories for test data and AGNI
RUN mkdir -p $FWL_DATA /opt/julia_depot

# Download required runtime data (from Zenodo, etc.)
# This ensures tests can run offline without downloading during execution
RUN cd /opt/proteus && \
    python -c "from proteus.utils.data import download_exoplanet_data, download_massradius_data; download_exoplanet_data(); download_massradius_data()"

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
