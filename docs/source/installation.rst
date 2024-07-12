Installation
==================================

These instructions will guide you through the typical installation process.
The setup is written for MacOS and Linux. Depending on your system settings and installed libraries your procedure may differ. 
If one or more of the steps below do not work for you we encourage you to first check the :doc:`troubleshooting` page. If that does not help you further, please contact the developers (see :doc:`contact`).


System configuration (MacOS)
--------------------------------

1. Open the terminal to install the developer tools 
      
    .. code-block:: console

        $  xcode-select --install
   
2. Install ``FORTRAN NetCDF`` library via the most appropriate method for you


    .. code-block:: console

        $   brew install netcdf  
        $   brew install netcdf-fortran    
    
    OR   

    .. code-block:: console

        $   sudo port install netcdf-fortran +gcc8   


System configuration (Linux)
--------------------------------
1. Install ``FORTRAN NetCDF`` via your package manager (e.g...)
    .. code-block:: console

        $   sudo apt install libnetcdff-dev


Python environment
--------------------------------
1 Download the appropriate Miniconda installer from https://docs.conda.io/en/latest/miniconda.html#id36

2 Create a conda environment for PROTEUS:

    .. code-block:: console
    
        $   conda create -n proteus python=3.12   
        $   conda activate proteus
        $   conda install matplotlib numpy pandas scipy sympy natsort ipykernel pip tomlkit
        $   conda install conda-forge::f90nml
        $   conda install conda-forge::netcdf4


Download the framework
--------------------------------
1. Register your public SSH key with Github:
    -  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys
    -  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
    -  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
    -  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection

2. Download PROTEUS base
    
    .. code-block:: console

        $   git clone git@github.com:FormingWorlds/PROTEUS.git

3. Enter into PROTEUS folder and ensure that submodules are up to date
    
    .. code-block:: console

        $   cd PROTEUS
        $   git submodule update --init --recursive

4. Setup atmosphere model (**JANUS**)
    
    1. Extract radiative transfer code (**SOCRATES**)

        .. code-block:: console

            $   cd JANUS/rad_trans/socrates_code/
            $   tar --strip-components 1 -xvf ../socrates_2306_trunk_r1403.tar.xz -C ./
            $   cp -f ../build_code_modified build_code

    2. Overwrite the ``Mk_cmd`` file with the right setup for your machine
        
        .. code-block:: console

            $   cp -rf ../Mk_cmd_SYSTEM make/Mk_cmd    
        
        The available options are:

        .. code-block:: console

            $   cp -rf ../Mk_cmd_MAC_INTEL make/Mk_cmd

        OR

        .. code-block:: console

            $   cp -rf ../Mk_cmd_MAC_APPLESILICON make/Mk_cmd

        OR

        .. code-block:: console

            $   cp -rf ../Mk_cmd_AOPP_CLUSTER make/Mk_cmd
            
        The command ``nf-config`` might be helpful if none of these options work for you.

    3. Setup SOCRATES

        .. code-block:: console
        
            $   ./build_code
            $   type ksh >/dev/null 2>&1 ||  sed -i 's/ksh/bash/g' sbin/*
            $   cd ../../../

5. Setup atmosphere kinetics model (**VULCAN**)

    .. code-block:: console

        $   cd VULCAN/fastchem_vulcan
    
    - On MacOS you will need to edit ``make.globaloptions`` to reflect a GNU-compatible ``g++`` executable, not the Apple one (see :doc:`troubleshooting` if the next step results in an error)
        
    .. code-block:: console

        $   make
        $   cd ../../

6. Setup stellar evolution model (**MORS**)

    .. code-block:: console

        $   cd Mors 
        $   wget http://www.astro.yale.edu/demarque/fs255_grid.tar.gz
        $   tar -xvf fs255_grid.tar.gz
        $   pip install -e .
        $   cd ../
    
7. Setup numerical computing library (**PETSc**)
    
    1. Configure step
        .. code-block:: console

            $   cd petsc
            $   ./configure --with-debugging=0 --with-fc=0 --with-cxx=0 --download-sundials2 --download-mpich --download-f2cblaslapack --COPTFLAGS="-g -O3" --CXXOPTFLAGS="-g -O3"
            
    2. Run the exact ``make all`` command provided at the end of the configure step
    
    3. Run the exact ``make check`` command provided at the end of the ``make all`` step

    4. Return to PROTEUS directory
    
        .. code-block:: console

            $   cd ../

8. Setup environment variables

    - This can be done using the PROTEUS environment file

        .. code-block:: console

            $   source PROTEUS.env

    - **IF** you want to be able to start PROTEUS immediately from a new shell every time, add ``source PROTEUS.env`` (and potentially ``conda activate proteus``) to your shell rc file.

9. Setup interior evolution model (**SPIDER**)

    .. code-block:: console

        $   cd SPIDER
        $   make clean
        $   make -j
        $   make test      # accept all default values when prompted
        $   cd ..

10. Done!


Step-by-step (optional modules)
--------------------------------

- Radiative-convective atmosphere model (**AGNI**)

    - Installation steps can be found at the `AGNI wiki <https://nichollsh.github.io/AGNI/dev/setup/>`_. They are also reproduced below.

    1. Setup Julia
        .. code-block:: console 

            $ curl -fsSL https://install.julialang.org | sh

    2. Clone the model 
        .. code-block:: console 

            $ git clone git@github.com:nichollsh/AGNI.git 
            $ cd AGNI 

    3. Setup SOCRATES 
        .. code-block:: console 

            $ source get_socrates.sh
    
    4. Build AGNI 
        .. code-block:: console 

            $ julia -e 'using Pkg; Pkg.activate("."); Pkg.build()'

    5. Go back to the PROTEUS directory 
        .. code-block:: console 

            $ cd ../

    - Consult the AGNI wiki if you encouter issues.

