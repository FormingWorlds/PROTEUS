#!/bin/bash
# Generate example from model output

# Check vars
if [[ -z $PROTEUS_DIR ]]; then
    echo "The PROTEUS_DIR variable has not been set."
    exit 1
fi
if [[ -z $1 ]]; then
    echo "You must provide the name of the target output folder."
    exit 1
fi

# Inform
echo "Making example of '$1'"

# Paths
EXA_DIR="$PROTEUS_DIR/examples/$1/"
OUT_DIR="$PROTEUS_DIR/output/$1/"

# Check paths
if [ ! -d "$OUT_DIR" ]; then
  echo "$OUT_DIR does not exist."
  exit 1
fi
if [ -d "$EXA_DIR" ]; then
  rm -r $EXA_DIR
fi

# Make example folder
mkdir $EXA_DIR

# Prepare to copy
shopt -s nullglob
cd $OUT_DIR

# Copy plots
for i in *.png *.pdf; do
    if [ -f "$i" ]; then
        cp $i $EXA_DIR
    fi
done

# Copy logs
for i in proteus*.log; do
    cp $i $EXA_DIR
done

# Copy data
cp runtime_helpfile.csv $EXA_DIR
cp init_coupler.toml $EXA_DIR
