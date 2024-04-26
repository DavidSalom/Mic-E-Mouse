#!/bin/zsh


while [[ true ]]
do

  sched_pid=$(<c_pid)
  kill $sched_pid 
  kill $sched_pid 

  echo "waiting for interval -- running from 8pm to 7am - signal"
  current_epoch=$(date +%s)
  target_epoch=$(date -d 'tomorrow 07:00' +%s)
  #target_epoch=$(date  "+%s" -d "5 second")
  sleep_seconds=$(( $target_epoch - $current_epoch ))

  if [[ $sleep_seconds -ge 0 ]]
  then
    echo "sleeping for $sleep_seconds seconds"
    sleep $sleep_seconds | pv -t
  fi

done
