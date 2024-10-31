#!/bin/bash
# Generate example from model output

# Check vars
if [[ -z $1 ]]; then
    echo "You must provide the name of the target output folder."
    exit 1
fi

# Inform
echo "Making example of '$1'"

# Paths
EXA_DIR="examples/$1/"
OUT_DIR="output/$1/"

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

# Copy files of a given type
shopt -s nullglob
for ext in "pdf" "png" "jpg" "toml"; do

  if [[ -n $(echo $OUT_DIR/*.$ext) ]]; then
    for f in "$OUT_DIR/*.$ext"; do
      cp $f $EXA_DIR
    done
  fi

done

# Copy PROTEUS logs
for f in "$OUT_DIR/proteus*.log"; do
  cp $f $EXA_DIR
done

# Copy other specific files
files=("runtime_helpfile.csv" "status")
for f in ${files[@]}; do
  cp "$OUT_DIR/$f" "$EXA_DIR/$f"
done

echo "Done"
