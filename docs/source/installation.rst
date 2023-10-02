Installation
=====

These instructions will guide you through the typical installation process. The setup is written primarily for MacOS (ARM: M1/M2 Macs, or Intel processors) and has been tested on Linux. Depending on your system settings and installed libraries your procedure may differ, however. If one or more of the steps below do not work for you we encourage you to first check the :ref:`troubleshooting` section, which lists many of the issues previously encountered. If that does not help you further, please contact the developers (see :doc:`contact`).

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
                
        * Latest tested version: *socrates_2306_trunk_r1403.tar.xz*

    * Interior dynamics: **SPIDER** 
        
        * URL: https://github.com/djbower/spider

    * Atmospheric chemistry: **VULCAN**
        
        * URL: https://github.com/exoclime/VULCAN/

    * Scientific computing library: **PETSc**
        
        * URL: https://gitlab.com/petsc/petsc

    * Scientific application test utility: **SciATH**
        
        * URL: https://github.com/sciath/sciath

Step-by-step (core modules)
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
                    
                    $   pip3 install matplotlib pandas netcdf4 matplotlib numpy pandas scipy sympy natsort netCDF4
                
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
                    $   conda install -c conda-forge f90nml netcdf4
            
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
            $   tar --strip-components 1 -xvf ../socrates_2306_trunk_r1403.tar.xz -C ./
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

    7. Setup Mors

        .. code-block:: console

            $   cd Mors 
            $   wget http://www.astro.yale.edu/demarque/fs255_grid.tar.gz
            $   tar -xvf fs255_grid.tar.gz
            $   cd ../
        
    8. Setup PETSc
        
        .. code-block:: console

            $   cd petsc
            $   ./configure --with-debugging=0 --with-fc=0 --with-cxx=0 --download-sundials2 --download-mpich --COPTFLAGS="-g -O3" --CXXOPTFLAGS="-g -O3"
                
        * Run the exact ``make all`` command provided at the end of the configure step
        
        * Run the exact ``make check`` command provided at the end of the ``make all`` step
        
        .. code-block:: console

            $   cd ../

    9. Setup environment variables
                
        * Only **IF** ``python`` has been installed via the ``conda`` route: 

            .. code-block:: console

                $   conda activate proteus
        
        * Setup the PROTEUS environment

            .. code-block:: console

                $   source PROTEUS.env

        * **IF** you want to be able to start PROTEUS immediately from a new shell every time, add ``source PROTEUS.env`` (and potentially ``conda activate proteus``) to your ``.zshrc`` (ARM) / ``.bash_profile`` (Intel)

    10. Setup SPIDER

        .. code-block:: console

            $   cd SPIDER
            $   make clean
            $   make -j
            $   make test      # accept all default values when prompted
            $   cd ..

**Done!**

Troubleshooting
----------------

This section includes troubleshooting advice for common errors. Each entry is labelled with the platform(s) typically affected. If you encounter errors or other issue that you cannot solve via the standard step-by-step guide or the advice below contact the developers (see :doc:`contact`).

* MacOS: PETSc tests error

    * Error when running the PETSc tests, looking like something along the lines of:
    
    .. code-block:: console
    
        Fatal error in PMPI_Init_thread: Other MPI error, error stack:
        MPIR_Init_thread(467)..............:
        MPID_Init(177).....................: channel initialization failed
        MPIDI_CH3_Init(70).................:
        MPID_nem_init(319).................:
        MPID_nem_tcp_init(171).............:`
        MPID_nem_tcp_get_business_card(418):
        MPID_nem_tcp_init(377).............: gethostbyname failed, localhost (errno 3)
    

    * This is actually a network configuration issue. To fix it, you need to add the following to ``/etc/hosts``, where`computername` is your hostname:    

    .. code-block:: console

        127.0.0.1   computername.local  
        127.0.0.1   computername

    * And then also enable Remote Login in your Sharing settings and add your user to the 'allowed access' list.

* All: PETSc complains about being in the wrong directory

    * Firstly, check that you are in the correct directory when running ``make`` or ``./configure``. If you are, then this could be caused by the environment variable ``PETSC_DIR`` remaining set after a previous PETSc installation. Run ``unset PETSC_DIR`` and try again.

* All: SPIDER can't find PETSc

    * Have you sourced ``PROTEUS.env``? If yes, check that the variables ``PETSC_ARCH`` and ``PETSC_DIR`` in that file are correct for your system.

* MacOS: The FastChem code distributed with VULCAN won't compile 

    * With the new Apple Silicon hardware (M1/M2), the option ``-march=native`` sometimes causes issues. In order to avoid this, you need to make sure to use the GNU version of ``g++``, not the Apple one. The Apple one located at ``/usr/bin/gcc`` is actually a wrapped around ``clang``. We found that using the Homebrew version located at ``/opt/homebrew/bin/`` works well. To fix this error, find out which ``gcc`` version homebrew installed (``ls /opt/homebrew/bin/gcc-*``), and edit the file ``make.globaloptions`` in the FastChem directory to use, e.g. ``g++-12`` or ``g++-13`` instead of ``g++``.

* Linux: ``ksh`` not found when running SOCRATES

    * Most Linux distributions do not come with ``ksh`` installed, while MacOS seems to. If you get an error relating to ``ksh`` not being found, check that you did all of the installation steps. One step under 'Setup SOCRATES' involves replacing ``ksh`` with ``bash`` in all of the SOCRATES executables.

* MacOS: Python / netCDF error ``Library not loaded: '@rpath/libcrypto.3.dylib'``

    * Create a symlink in the local Python installation (here shown for ``bash`` terminal). See https://pavcreations.com/dyld-library-not-loaded-libssl-1-1-dylib-fix-on-macos/

    .. code-block:: console

        $   brew install openssl

    * Follow the instructions at the end of the ``openssl`` installation (replace ``USERNAME`` with your own system username):

    .. code-block:: console

        $   echo 'export PATH="/usr/local/opt/openssl@3/bin:$PATH"' >> /Users/USERNAME/.bash_profile  
        $   echo 'export LDFLAGS="-L/usr/local/opt/openssl@3/lib"' >>/Users/USERNAME/.bash_profile  
        $   echo 'export CPPFLAGS="-I/usr/local/opt/openssl@3/include"' >>/Users/USERNAME/.bash_profile
        $   ln -s /usr/local/opt/openssl/lib/libcrypto.3.dylib /Users/USERNAME/opt/anaconda3/envs/proteus/lib/python3.10/site-packages/netCDF4/../../../
        $   ln -s /usr/local/opt/openssl/lib/libssl.3.dylib /Users/USERNAME/opt/anaconda3/envs/proteus/lib/python3.10/site-packages/netCDF4/../../../

* MacOS: Python error ``ModuleNotFoundError: No module named 'yaml'`` despite ``yaml`` being installed via ``conda``

    .. code-block:: console

        $   python -m pip install pyyaml

* MacOS: If the `SOCRATES make` routine complains about missing ``ifort`` compilers
    
    * Install Intel compilers from https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html
    * First Intel® oneAPI Base Toolkit
    * Then Intel® oneAPI HPC Toolkit
    * Follow the instructions that are provided after the installation to set the locations of ``ifort`` in your environment

* MacOS: One of the following errors during PETSC configuration or compilation steps
``"This header is only meant to be used on x86 and x64 architecture"``

``#error "This header is only meant to be used on x86 and x64 architecture"``

    * Follow **Option A** in the step-by-step guide to (re-)install ``python``

* MacOS: ``ModuleNotFoundError: No module named '_tkinter'``
    
    * Install ``tkinter`` packageu using ``brew``: 
    
    .. code-block:: console
        
        $   brew install python-tk

* MacOS: In the terminal or SourceTree ``Error: Permission denied (publickey)``
    
    * Your ssh key is out of date, follow:

    1.  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys
    
    2.  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
    
    3.  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
    
    4.  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection


* MacOS: An error during the SOCRATES compilation: 
   `` ld: unsupported tapi file type '!tapi-tbd' in YAML file '/Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk/usr/lib/libSystem.tbd' for architecture arm64``

      * There is an issue with your ``ld``, potentially caused by an existing installation of ``anaconda``
      * Delete all traces of ``anaconda`` by following the steps at https://docs.anaconda.com/free/anaconda/install/uninstall/
      * Install ``python`` via ``brew`` (see above in the main installation instructions)


* MacOS: One of the following errors during the SOCRATES compilation: 

``clang (LLVM option parsing): Unknown command line argument '-x86-pad-for-align=false'.  Try: 'clang (LLVM option parsing) --help'``

``clang (LLVM option parsing): Did you mean '--x86-slh-loads=false'?``

        * There is an issue with your compiler, either the standard Apple ``clang`` or ``gcc`` installed by ``brew``
        * Follow the steps provided at https://stackoverflow.com/questions/72428802/c-lang-llvm-option-parsing-unknown-command-line-argument-when-running-gfort
      
        .. code-block:: console

            $   sudo rm -rf /Library/Developer/CommandLineTools
            $   sudo xcode-select --install





