#!/bin/bash
#
#



inputs=$(cat /proc/bus/input/devices | grep -A5 -ne "\"Razer Razer Viper 8KHz\"" |
  cat /proc/bus/input/devices | grep -A5 -ne "\"Razer Razer Viper 8KHz\"" | sed '/Handlers=.* mouse[0-9]/!d' | sed -E 's/(.*)(mouse.*)/\2/g')

echo "$inputs"
