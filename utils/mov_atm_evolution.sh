# Where are the image folders?
IMAGE_DIR_BASE=/Users/tim/runs/coupler_tests/200501

# Where to save the movie files
MOVIE_DIR=/Users/tim/runs/coupler_tests/200501

# Frames per second
FPS=24

# Movie length [s]
ML=60

# Loop through simulation branches
# H2 H2O
for BATCH in H2O H2 CH4 CO2 CO N2 O2; do

    # Absolute directory of where images are stored
    # IMAGE_DIR=${IMAGE_DIR_BASE}/${BATCH}/${MOVIE_NAME}
    IMAGE_DIR=${IMAGE_DIR_BASE}/${BATCH}/mov_figs
    # MOVIE_NAME=${PWD##*/}

    # Go the base directory of the folders which contain the images
    cd ${IMAGE_DIR}
    pwd

    #for DIR in $(ls -d r*/); do
    # for FIELD in $BATCH; do

    # Get movie name from folder name of the images
    # MOVIE_NAME=${DIR%%/}
    # MOVIE_NAME=${PWD##*/}
    MOVIE_NAME=${BATCH}

    # Calculate shown images per second, such that movies are always $ML seconds long
    IMAGE_NO=$(ls $IMAGE_DIR | wc -l)
    IPS=$(echo "$IMAGE_NO/$ML" | bc)

    # Inform user about the input files
    echo "Use the following files as input:"
    ls $IMAGE_DIR/*.png

    # Process the files and create movie
    ffmpeg -framerate $IPS -pattern_type glob -i "$IMAGE_DIR/*.png" -vsync 2 -vf scale=1504:1232 -y -c:v libx264 -r $FPS -pix_fmt yuv420p -profile:v baseline -level 3.0 -movflags +faststart -preset veryslow -crf 18 ${MOVIE_DIR}/mov_${MOVIE_NAME}.mp4 #_${FIELD}
    # done
done

# Errors and solutions
# [image2 demuxer @ 0x7faac200aa00] Error setting option framerate to value 0.
# /Users/tim/bitbucket/pcd_couple-interior-atmosphere/atm_rad_conv/output/mov_figs/trpp_H2O/*.png: Invalid argument
# --> https://superuser.com/questions/602950/problems-with-frame-rate-on-video-conversion-using-ffmpeg-with-libx264

# Lossless video: https://trac.ffmpeg.org/wiki/Encode/H.264
