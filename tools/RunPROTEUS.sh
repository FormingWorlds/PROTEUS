#!/bin/bash

# Script to run PROTEUS using screen

echo "Start RunPROTEUS"

# Check if the required globals are set
if [[ -z $COUPLER_DIR ]]; then
    echo "    ERROR: Required global variables have not been set"
    echo "           Did you source the env file yet?"
    sleep 3.0 
    exit 1
fi


# Check if the required arguments have been passed
if [ -z "$1" ] || [ -z "$3" ]
then
    echo "    ERROR: Config file or alias provided" 
    echo "    First argument:   config file     (string)"
    echo "    Second argument:  screen alias    (string)"
    echo "    Third argument:   detach?         (y or n)"
    sleep 3.0 
    exit 1
else
    # Set variables
    CFGFILE="$1"
    ALIAS="$2"
    DETACH=$(echo "$3" | tr -d ' ' | tr '[:upper:]' '[:lower:]' | cut -c1-1)  # strip spaces, covert to lowercase, get first char
    LOGFILE="$COUPLER_DIR/output/$ALIAS.log"
    EXECUTABLE="$COUPLER_DIR/proteus.py"

    # Clear dead screens
    screen -wipe > /dev/null

    # Check if it's already running
    if [[ $(screen -ls | grep -E "$ALIAS[[:blank:]]") ]]; then
        echo "    ERROR: PROTEUS is already running under this alias"
        screen -ls | grep -E "$ALIAS[[:blank:]]"
        sleep 3.0
        exit 1
    fi

    # Setup log file
    echo "    Config path   = '$CFGFILE' "
    echo "    Log file path = '$LOGFILE' "
    echo "    Screen alias  = '$ALIAS' "

    rm -f $LOGFILE
    touch $LOGFILE

    # Dispatch screen session with PROTEUS inside
    echo "    Dispatching screen session..."
    COMMAND="python $EXECUTABLE --cfg_file $CFGFILE"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # MacOS does not support the -Logfile flag
        config="log on
        logfile $LOGFILE";
        config_file="/tmp/$ALIAS.screenconf"
        rm -f "$config_file"
        echo "$config" > "$config_file"
        if [[ "$DETACH" == "y" ]]; then 
            screen -S $ALIAS -L -d -m -c "$config_file" bash -c "$COMMAND" 
        else 
            screen -S $ALIAS -L -c "$config_file" bash -c "$COMMAND" 
        fi
        
    else
        # Linux
        if [[ "$DETACH" == "y" ]]; then 
            screen -S $ALIAS -d -m -L -Logfile $LOGFILE bash -c "$COMMAND" 
        else 
            screen -S $ALIAS -L -Logfile $LOGFILE bash -c "$COMMAND" 
        fi
    fi
    
    # Done?
    echo "        (detached)"
    exit 0
fi
