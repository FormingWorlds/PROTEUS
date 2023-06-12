#!/bin/bash

# Script to delete socrates files in the current working directory if they 
# happen to be left over (e.g. if a run is killed early).

rm profile.*
rm radiation_code.lock
rm currentsw.*
