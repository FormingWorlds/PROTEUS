#!/bin/bash

# Script to run PROTEUS using screen

echo "Start RunPROTEUS"

# Check if the required globals are set
if [[ -z $COUPLER_DIR ]]; then
    echo "ERROR: Required global variables have not been set"
    echo "       Did you source the env file yet?"
    sleep 1.0 
    exit 1
fi
if [[ -z $RAD_DIR ]]; then 
    echo "ERROR: The RAD_DIR variable has not been set"
    sleep 1.0 
    exit 1 
fi


# Check if the required arguments have been passed
if [ -z "$1" ] || [ -z "$3" ]
then
    echo "ERROR: Config file or alias not provided" 
    echo "       First argument:   config file     (string)"
    echo "       Second argument:  screen alias    (string)"
    echo "       Third argument:   resume?         (y or n)"
    echo "       Fourth argument:  detach?         (y or n)"
    sleep 1.0 
    exit 1
else
    # Set variables
    CFGFILE="$1"
    ALIAS="$2"
    RESUME=$(echo "$3" | tr -d ' ' | tr '[:upper:]' '[:lower:]' | cut -c1-1)  # strip spaces, covert to lowercase, get first char
    DETACH=$(echo "$4" | tr -d ' ' | tr '[:upper:]' '[:lower:]' | cut -c1-1)  # strip spaces, covert to lowercase, get first char
    EXECUTABLE="$COUPLER_DIR/proteus.py"

    # Clear dead screens
    screen -wipe > /dev/null

    # Check if it's already running
    if [[ $(screen -ls | grep -E "$ALIAS[[:blank:]]") ]]; then
        echo "    ERROR: PROTEUS is already running under this alias"
        screen -ls | grep -E "$ALIAS[[:blank:]]"
        sleep 1.0
        exit 1
    fi

    # Setup paths 
    echo "    Config path   = '$CFGFILE' "
    echo "    Screen alias  = '$ALIAS' "

    # Dispatch screen session with PROTEUS inside
    echo "    Dispatching screen session..."

    COMMAND="python $EXECUTABLE --cfg $CFGFILE"
    if [[ "$RESUME" == "y" ]]; then 
        COMMAND="$COMMAND --resume"
    fi 

    if [[ "$DETACH" == "y" ]]; then 
        screen -S $ALIAS -d -m bash -c "$COMMAND" 
    else 
        screen -S $ALIAS bash -c "$COMMAND" 
    fi
    
    # Done?
    echo "    (detached)"
    exit 0
fi
