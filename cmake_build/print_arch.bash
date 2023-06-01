#!/bin/bash
uname -s | awk '{print tolower($0)}'
