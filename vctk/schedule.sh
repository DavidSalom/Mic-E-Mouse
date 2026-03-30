#!/bin/zsh

# Validate mouse_id
mouse_id=$1
[[ -z "$mouse_id" ]] && mouse_id=0

# Make sure expected mouse is connected
if [ $mouse_id -eq 0 ];
then
    mouse_name="Razer Razer Viper 8KHz"
elif [ $mouse_id -eq 1 ];
then
    mouse_name="Logitech G502 HERO"
else
  echo "Invalid mouse_id: $mouse_id, expected 0 or 1"
  exit
fi
cmd=""
mouse_device=$(cat /proc/bus/input/devices | grep -e \"$mouse_name\")
if [ -z "$mouse_device" ];
then
    echo "Mouse not found: $mouse_name"
    exit
fi

path_file=$2
if [[ ! -z "$path_file" && ! -f "$path_file" ]] 
then
  echo "nonexistent file: $path_file"
  exit

else

  zmodload zsh/mapfile
  pf_paths=( "${(f)mapfile[$path_file]}" )

  #
  # NOTE: you need to path_file paths to NOT start with vctk
  #

  #pf_paths=$list_flac
  declare -a q_pf

  for item in $pf_paths
  do
      i_ext=$item:t:e

      if [[ ! "-f ../$item" || "$i_ext" -ne "flac" ]]
      then
        echo "BAD FILE: "
      else
        #echo $i_ext
        #accept_path=$(soxi -D $(echo "$item" | tr -d ' '))
        accept_path=$(echo "$item" | tr -d ' ')

        q_pf+=( $accept_path )
      fi
      #echo $q_pf
  done

  n_flac=$#q_pf
  n_all_flac=$#q_pf
  n_comp=0

  echo "using input path file: $path_file with $n_flac valid FLAC files"
fi


# Default start and end times
default_start_time="21:00" # 9 PM
default_end_time="08:00" # 8 AM next day
default_num_days=1 # 1 day

# Read start time from user or set to default
read "start_time?Enter start time (HH:MM) or press Enter for default ($default_start_time): "
[[ -z "$start_time" ]] && start_time=$default_start_time

# Read end_date from user or set to default, end_date is a number of days from today, default is 1
read "end_date?Enter end date (number of days from today) or press Enter for default (1). (Note: if setting on Weekends, the value should be 3): "
[[ -z "$end_date" ]] && end_date=$default_num_days

# Read end time from user or set to default
read "end_time?Enter end time next day (HH:MM) or press Enter for default ($default_end_time): "
[[ -z "$end_time" ]] && end_time=$default_end_time

# Convert current time, start time and end time to minutes since 00:00
current_time_minutes=$((10#$(date +%H)*60 + 10#$(date +%M)))
start_time_minutes=$((10#${start_time%:*}*60 + 10#${start_time#*:}))
end_time_minutes=$((10#${end_time%:*}*60 + 10#${end_time#*:}))
end_time_minutes=$(( end_time_minutes + 24*60*end_date )) # Adjust end_time_minutes by end_date * 24 hours * 60 minutes

# Calculate sleep_time and record_time
sleep_time=$((start_time_minutes - current_time_minutes))
record_time=$((end_time_minutes - start_time_minutes))

# Tell user what is going to happen
echo "==="
echo "Current time: $(date +%H:%M)"
echo "Start time: $start_time"
echo "End time: $end_time"
echo "Sleep time: $sleep_time minutes"
echo "Record time: $record_time minutes"
echo "Mouse ID: $mouse_id"
echo "==="
echo "Sleeping for $sleep_time minutes, then recording for $record_time minutes..."
# Assuming mouse_id is provided as the first script argument $1
sleep ${sleep_time}m && timeout ${record_time}m ./create.sh $mouse_id
