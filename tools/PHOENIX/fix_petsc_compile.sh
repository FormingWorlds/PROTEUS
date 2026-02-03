# Go into the petsc directory
cd petsc

# Detect your shell config file
if [[ "$SHELL" == */zsh ]]; then
  MY_RC="$HOME/.zshrc"
elif [[ "$SHELL" == */bash ]]; then
  MY_RC="$HOME/.bash_profile"
else
  echo "Could not detect Zsh or Bash. Please add lines manually."
fi

# Append the correct environment variables to your shell config file
if [[ -n "$MY_RC" ]]; then
  echo "" >> "$MY_RC"
  echo "# PETSc/PROTEUS Environment Variables" >> "$MY_RC"
  echo 'export SDKROOT=$(xcrun --show-sdk-path)' >> "$MY_RC"
  echo 'export CC=mpicc' >> "$MY_RC"
  echo 'export CXX=mpicxx' >> "$MY_RC"
  echo 'export FC=mpifort' >> "$MY_RC"
  echo 'export F77=mpifort' >> "$MY_RC"

  echo "Successfully added lines to $MY_RC"
  echo "Run this command now to activate them: source $MY_RC"
fi

# Run the configure command
cd petsc
./configure \
   PETSC_ARCH=arch-darwin-c-opt \
   CC=mpicc \
   CXX=mpicxx \
   FC=mpifort \
   F77=mpifort \
   LDFLAGS="-L/opt/homebrew/lib -Wl,-w" \
   --with-debugging=0 \
   --with-cxx-dialect=14 \
   --download-sundials2=1

# 3. Once that finishes successfully, build and test it
make PETSC_DIR=$(pwd) PETSC_ARCH=arch-darwin-c-opt all
make PETSC_DIR=$(pwd) PETSC_ARCH=arch-darwin-c-opt check
