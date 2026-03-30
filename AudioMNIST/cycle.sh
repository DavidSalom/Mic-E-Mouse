#!/bin/zsh

while true;
do
  DATE=`date "+%T" | cut -d' ' -f4`
  t_date="20:00:01"
  echo "sleeping: $DATE until $t_date"
  #if [[ $DATE == "08:00:00" ]]
  if [[ $DATE == "$t_date" ]]
  then

    timeout 11h ./schedule_test.sh &
    wait $!
    
    sched_pid=$(<c_pid)
    sched_pid_c=$(<c_pid_c)
    kill -9 $sched_pid 
    kill -9 $sched_pid_c 
    sleep 1s
  fi
  sleep 1s
done

