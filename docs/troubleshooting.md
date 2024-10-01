# Troubleshooting

This section includes troubleshooting advice for common errors. Each
entry is labelled with the platform(s) typically affected. If you
encounter errors or other issue that you cannot solve via the standard
step-by-step guide or the advice below contact the developers (see
`contact`{.interpreted-text role="doc"}).

## GitHub: Cannot clone module

Have you added your SSH key to GitHub? See these pages for guidance:
* [Checking for existing SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys)
* [Generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
* [Adding a new SSH key to your Github account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
* [Testing your SSH connection](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection)

## MacOS: PETSc tests error

Error when running the PETSc tests, looking like something along the lines of:

```console
Fatal error in PMPI_Init_thread: Other MPI error, error stack:
MPIR_Init_thread(467)..............:
MPID_Init(177).....................: channel initialization failed
MPIDI_CH3_Init(70).................:
MPID_nem_init(319).................:
MPID_nem_tcp_init(171).............:`
MPID_nem_tcp_get_business_card(418):
MPID_nem_tcp_init(377).............: gethostbyname failed, localhost (errno 3)
```
This is actually a network configuration issue. To fix it, you need to add the following to `/etc/hosts`, where `computername` is your hostname:

```console
127.0.0.1   computername.local
127.0.0.1   computername
```
And then also enable Remote Login in your Sharing settings and add your user to the *allowed access* list.

## All: PETSc complains about being in the wrong directory

Firstly, check that you are in the correct directory when running `make` or `./configure`. If you are, then this could
be caused by the environment variable `PETSC_DIR` remaining set after a previous PETSc installation. Run `unset PETSC_DIR`
and try again.

## Linux: `ksh` not found when running SOCRATES

Most Linux distributions do not come with `ksh` installed, while MacOS seems to. If you get an error relating to `ksh` not being found, check that you did all of the installation steps.

One step under *Setup SOCRATES* involves replacing `ksh` with `bash` in all of the SOCRATES executables.

## MacOS: The FastChem code distributed with VULCAN won\'t compile

With the new Apple Silicon hardware (M1/M2), the option `-march=native` sometimes causes issues.

To avoid this, you need to make sure to use the GNU version of `g++`, not the Apple one. The Apple one located at `/usr/bin/gcc` is actually a wrapped around `clang`.

We found that using the Homebrew version located at `/opt/homebrew/bin/` works well.

To fix this error, find out which `gcc` version homebrew installed (`ls /opt/homebrew/bin/gcc-*`), and edit the file`make.globaloptions` in the FastChem directory to use, e.g. `g++-12` or `g++-13` instead of `g++`.

## MacOS: Python / netCDF error

Error: `Library not loaded: '@rpath/libcrypto.3.dylib'`

Create a symlink in the local Python installation (here shown for `bash` terminal). See
 <https://pavcreations.com/dyld-library-not-loaded-libssl-1-1-dylib-fix-on-macos/>

```console
brew install openssl
```
Follow the instructions at the end of the `openssl` installation (replace `USERNAME` with your own system username):

```console
echo 'export PATH="/usr/local/opt/openssl@3/bin:$PATH"' >> /Users/USERNAME/.bash_profile
echo 'export LDFLAGS="-L/usr/local/opt/openssl@3/lib"' >>/Users/USERNAME/.bash_profile
echo 'export CPPFLAGS="-I/usr/local/opt/openssl@3/include"' >>/Users/USERNAME/.bash_profile
ln -s /usr/local/opt/openssl/lib/libcrypto.3.dylib /Users/USERNAME/opt/anaconda3/envs/proteus/lib/python3.10/site-packages/netCDF4/../../../
ln -s /usr/local/opt/openssl/lib/libssl.3.dylib /Users/USERNAME/opt/anaconda3/envs/proteus/lib/python3.10/site-packages/netCDF4/../../../
```

## MacOS: Python error `ModuleNotFoundError: No module named 'yaml'`
    despite `yaml` being installed via `conda`

```console
python -m pip install pyyaml
```

## MacOS: Missing `ifort` compilers

If the [SOCRATES make]{.title-ref} routine complains about missing `ifort` compilers:

Install Intel compilers from <https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html>

First Intel® oneAPI Base Toolkit, then Intel® oneAPI HPC Toolkit

Follow the instructions that are provided after the
installation to set the locations of `ifort` in your
environment.

## MacOS: Errors during PETSc configuration or compilation

One of the following errors during PETSC configuration or compilation steps
- `"This header is only meant to be used on x86 and x64 architecture"`
- `#error "This header is only meant to be used on x86 and x64 architecture"`

Follow **Option A** in the step-by-step guide to (re-)install `python`

## MacOS: `ModuleNotFoundError: No module named '_tkinter'`

Install `tkinter` package using `brew`:


```console
brew install python-tk
```

## MacOS: `ImportError: Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running`

If you are connecting to a computer over ssh, make sure to enable X-forwarding. This can be done in your configuration file or as a command line parameter.

If you are using a Mac, install [XQuartz](https://www.xquartz.org/).

## MacOS: `Permission denied (publickey)`

You get this error In the terminal or SourceTree. This means your ssh key is out of date, follow:

1. <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys>
2. <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>
3. <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>
4. <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection>

## MacOS: YAML library error

Error: `ld: unsupported tapi file type '!tapi-tbd' in YAML file '/Library/Developer/CommandLineTools/SDKs/MacOSX13.sdk/usr/lib/libSystem.tbd' for architecture arm64`

There is an issue with your `ld`, potentially caused by an existing installation of `anaconda`.

Delete all traces of `anaconda` by following the steps at <https://docs.anaconda.com/free/anaconda/install/uninstall/>.

Install `python` via `brew` (see above in the main installation instructions)

## MacOS: Errors during the SOCRATES compilation

One of the following errors happen during the SOCRATES compilation:

1. `clang (LLVM option parsing): Unknown command line argument '-x86-pad-for-align=false'.  Try: 'clang (LLVM option parsing) --help'`
2. `clang (LLVM option parsing): Did you mean '--x86-slh-loads=false'?`

There is an issue with your compiler, either the standard Apple `clang` or `gcc` installed by `brew`.

Follow the steps provided at <https://stackoverflow.com/questions/72428802/c-lang-llvm-option-parsing-unknown-command-line-argument-when-running-gfort>


```console
sudo rm -rf /Library/Developer/CommandLineTools
sudo xcode-select --install
```

## MacOS: gfortran cannot find system libraries

Error: `ld library not found for -lsystem`

This usually occurs when your operating system has been updated, but Xcode is not on the latest version

To fix this, try updating or reinstalling Xcode.

Alternatively, try adding the following flag to the gfortran
compile and link steps:

```console
-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib/`.
```
