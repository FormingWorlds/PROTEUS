# Proteus
export COUPLER_DIR="$(pwd)"
export PATH=$COUPLER_DIR:$PATH
export PYTHONPATH=$COUPLER_DIR:$PYTHONPATH
export PATH=$COUPLER_DIR/utils:$PATH
export PYTHONPATH=$COUPLER_DIR/utils:$PYTHONPATH
echo "COUPLER_DIR = $COUPLER_DIR"

# Socrates and AEOLUS
export PYTHONPATH=$COUPLER_DIR/AEOLUS:$PYTHONPATH
source "$COUPLER_DIR/AEOLUS/rad_trans/socrates_code/set_rad_env"

# Spider
SPIDER_DIR=$COUPLER_DIR/SPIDER
export PETSC_DIR=$COUPLER_DIR/petsc-double
export PATH=$SPIDER_DIR:$PATH
export PATH=$SPIDER_DIR/py3:$PATH
export PYTHONPATH=$SPIDER_DIR/py3:$PYTHONPATH
export PYTHONPATH=$COUPLER_DIR/sciath:$PYTHONPATH
export PETSC_ARCH=arch-linux-c-opt               # EDIT THIS VARIABLE AS APPROPRIATE

# Vulcan
export PYTHONPATH=$COUPLER_DIR/VULCAN:$PYTHONPATH

