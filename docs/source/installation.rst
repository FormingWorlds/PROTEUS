Installation
=====

These instructions will guide you through the typical installation process. The setup is written primarily for MacOS (ARM: M1/M2 Macs, or Intel processors) and has been tested on Linux. Depending on your system settings and installed libraries your procedure may differ, however. If one or more of the steps below do not work for you we encourage you to first check the :doc:`troubleshooting` section, which lists many of the issues previously encountered. If that does not help you further, please contact the developers (see :doc:`contact`).

Software dependencies
----------------

Read access to the following repositories:

    * Coupler framework: **PROTEUS**
        
        * URL: https://github.com/FormingWorlds/PROTEUS

    * Radiative-convective scheme: **AEOLUS** 
        
        * URL: https://github.com/FormingWorlds/AEOLUS/

    * Radiation transport: **SOCRATES** 
        
        * Main development URL: https://code.metoffice.gov.uk/trac/socrates
        
        * Contact: james.manners@metoffice.gov.uk
        
        * Obtain the SOCRATES source code from: https://simplex.giss.nasa.gov/gcm/ROCKE-3D/ (Latest released version of SOCRATES code)
        
        * Latest tested version: *socrates_2211.tar.xz*

    * Interior dynamics: **SPIDER** 
        
        * URL: https://github.com/djbower/spider

    * Atmospheric chemistry: **VULCAN**
        
        * URL: https://github.com/exoclime/VULCAN/

    * Scientific computing library: **PETSc**
        
        * URL: https://gitlab.com/petsc/petsc

    * Scientific application test utility: **SciATH**
        
        * URL: https://github.com/sciath/sciath

Step-by-step guide
----------------

1. Install dependencies

    * MacOS: Ensure installation of command line developer tools
    
    * Open Xcode application and install
      
      .. code-block:: console

         $  xcode-select --install`
   
    * Install ``FORTRAN NetCDF`` library via the most appropriate method for you

      .. code-block:: console

        $   brew install netcdf  
        $   brew install netcdf-fortran    
        
      OR   

      .. code-block:: console
        
        $   sudo port install netcdf-fortran +gcc8   
        
      OR   

      .. code-block:: console
        
        $   sudo apt install libnetcdff-dev
    
    * Set up a Python environment:
         
         * Option A (*recommended*): using the `brew` package manager
            
            * The following steps assume ``brew`` (if not, follow: https://brew.sh/) is installed on your system.
            
            * Delete all traces of a potential Anaconda package manager installation from your system. 
                
                * To do this, follow the steps at https://docs.anaconda.com/free/anaconda/install/uninstall/
                
                * Delete all Anaconda-related entries from your ``~/.bash_profile`` (Intel) or ``~/.zshrc`` (ARM)
            
            * Install Python via ``brew``:
                
                .. code-block:: console 
                    
                    $   brew install python
                
                * Update to the latest stable version:
                
                .. code-block:: console
                    
                    $   brew upgrade python
                
                * Install ``tkinter``: 
                
                .. code-block:: console
                    
                    $   brew install python-tk@3.11
                
                * Refresh your shell:
                    * ARM:
                    
                    .. code-block:: console
                        
                        $   source ~/.zsrhrc
                    
                    * Intel:
                    
                    .. code-block:: console
                    

                        $   source ~/.bash_profile
                
                * Install all other necessary packages: 
                
                .. code-block:: console
                    
                    $   pip3 install matplotlib pandas netcdf4 matplotlib numpy pandas scipy sympy natsort
                
                * Make the new Python version the system default (check what `brew` tells you during/after the `brew install python` step), by adding the following to your:
                    
                    * ``~/.zsrhrc`` (ARM):
                    
                    .. code-block:: console
                        
                        $   export PATH="/opt/homebrew/opt/python/libexec/bin:$PATH"
                    
                    * ``~/.bash_profile`` (Intel): 
                    
                    .. code-block:: console
                        
                        $   export PATH="/usr/local/opt/python/libexec/bin:$PATH"
         
         * Option B: Using the ``anaconda`` package manager (be careful, this potentially breaks the PETSc installation on ARM)
            
            * Install ``conda``:
                
                * Download the appropriate Miniconda installer from https://docs.conda.io/en/latest/miniconda.html#id36
                
                * Create a conda environment for PROTEUS:
                
                .. code-block:: console
                    
                    $   conda create -n proteus python=3.10.9   
                    $   conda activate proteus
                    $   conda install netcdf4 matplotlib numpy pandas scipy sympy natsort
                    $   conda install -c conda-forge f90nml
            
            * Refresh your shell:
                    
                    * ARM:
                    
                    .. code-block:: console
                        
                        $   source ~/.zsrhrc
                    
                    * Intel:
                    
                    .. code-block:: console
                        
                        $   source ~/.bash_profile
        
    * Register your public SSH key with Github:
        
        1.  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys
        
        2.  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
        
        3.  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
        
        4.  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection

3. Setup codes and modules in the following order (ignore the instructions provided in their own repositories)

    1. Download PROTEUS + submodules
        
        .. code-block:: console
                        
            $   git clone --recursive git@github.com:FormingWorlds/PROTEUS.git

    2. Enter into PROTEUS folder and ensure that submodules are up to date
        
        .. code-block:: console

            $   cd PROTEUS
            $   git submodule update --init --recursive

    3. Download and extract SOCRATES to the correct location
        
        .. code-block:: console

            $   cd AEOLUS/rad_trans/socrates_code/
            $   curl -L -o ../socrates_2211.tar.xz https://www.dropbox.com/sh/ixefmrbg7c94jlj/AAChibnZU9PRi8pXdVxVbdj3a/socrates_2211.tar.xz?dl=1
            $   tar --strip-components 1 -xvf ../socrates_2211.tar.xz -C ./
            $   cp -f ../build_code_modified build_code

    4. Overwrite the ``Mk_cmd`` file with the right setup for your machine
        
        .. code-block:: console

            $   cp -rf ../Mk_cmd_SYSTEM make/Mk_cmd    
        
        * The available options are:

        .. code-block:: console

            $   cp -rf ../Mk_cmd_MAC_INTEL make/Mk_cmd

        OR

        .. code-block:: console

            $   cp -rf ../Mk_cmd_MAC_APPLESILICON make/Mk_cmd

        OR

        .. code-block:: console

            $   cp -rf ../Mk_cmd_AOPP_CLUSTER make/Mk_cmd
            
        The command ``nf-config`` might be helpful if none of these options work for you.

    5. Setup SOCRATES

        .. code-block:: console
        
            $   ./build_code
            $   type ksh >/dev/null 2>&1 ||  sed -i 's/ksh/bash/g' sbin/*
            $   cd ../../../

    6. Setup VULCAN

        .. code-block:: console

            $   cd VULCAN/fastchem_vulcan
        
        * On MacOS you will need to edit ``make.globaloptions`` to reflect a GNU-compatible ``g++`` executable, not the Apple one (see :doc:`troubleshooting` if the next step results in an error)
            
        .. code-block:: console

            $   make
            $   cd ../../
        
    7. Setup PETSc
        
        .. code-block:: console

            $   cd petsc
            $   ./configure --with-debugging=0 --with-fc=0 --with-cxx=0 --download-sundials2 --download-mpich --COPTFLAGS="-g -O3" --CXXOPTFLAGS="-g -O3"
        
        * Make note of the value of ``PETSC_ARCH`` printed to ``stdout``.
        
        * Run the exact ``make all`` command provided at the end of the configure step
        
        * Run the exact ``make check`` command provided at the end of the ``make all`` step
        
        .. code-block:: console

            $   cd ../

    8. Setup environment variables
        
        * Edit the variable ``PETSC_ARCH`` in the file ``PROTEUS.env`` to reflect value provided by PETSc
        
        * Only **IF** ``python`` has been installed via the ``conda`` route: 

            .. code-block:: console

                $   conda activate proteus
        
        * Setup the PROTEUS environment

            .. code-block:: console

                $   source PROTEUS.env

        * **IF** you want to be able to start PROTEUS immediately from a new shell every time, add ``source PROTEUS.env`` to your ``.zshrc`` (ARM) / ``.bash_profile`` (Intel)

    9. Setup SPIDER

        .. code-block:: console

            $   cd SPIDER
            $   make clean
            $   make -j
            $   make test      # accept all default values when prompted
            $   cd ../

**Done!**

Troubleshooting
----------------

