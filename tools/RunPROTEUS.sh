#!/bin/bash

# Script to run PROTEUS using screen

# Check if the required globals are set
if [[ -z $COUPLER_DIR ]]; then
    echo "ERROR: Required global variables have not been set"
    echo "       Did you source the env file yet?"
    exit 1
fi


# Check if the required arguments have been passed
if [ -z "$1" ] || [ -z "$2" ]
then
    echo "Error: Config file or alias provided"
    echo "First argument:  config file"
    echo "Second argument: alias"
    exit 1
else
    # Set variables
    CFGFILE="$1"
    ALIAS="$2"
    LOGFILE="$COUPLER_DIR/output/$ALIAS.log"

    # Check if it's already running
    if [[ $(screen -ls | grep -E "$ALIAS[[:blank:]]") ]]; then
        echo "ERROR: PROTEUS is already running under this alias"
        exit 1
    fi

    # Setup log file
    echo "Config path   = '$CFGFILE' "
    echo "Log file path = '$LOGFILE' "
    echo "Screen alias  = '$ALIAS' "

    rm -f $LOGFILE
    touch $LOGFILE

    echo "Sleeping 3 seconds..."
    sleep 3

    # Dispatch screen session with PROTEUS inside
    echo "Dispatching screen session..."
    COMMAND="python proteus.py -cfg_file $CFGFILE"
    screen -S $ALIAS -L -Logfile $LOGFILE bash -c "$COMMAND" 

    # Done?
    echo "Detached"
    exit 0
fi
