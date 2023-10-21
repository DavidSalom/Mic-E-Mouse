#!/bin/zsh

#if [[ ! $(sudo echo 0) ]]; then exit; fi
pid_log=0

sigint_handler()
{
    kill -s SIGINT $pid_log
    exit
}

percentBar ()  { 
    local prct totlen=$((8*$2)) lastchar barstring blankstring;
    printf -v prct %.2f "$1"
    ((prct=10#${prct/.}*totlen/10000, prct%8)) &&
        printf -v lastchar '\\U258%X' $(( 16 - prct%8 )) ||
            lastchar=''
    printf -v barstring '%*s' $((prct/8)) ''
    printf -v barstring '%b' "${barstring// /\\U2588}$lastchar"
    printf -v blankstring '%*s' $(((totlen-prct)/8)) ''
    printf -v "$3" '%s%s' "$barstring" "$blankstring"
}

IFS=$'\n'
list_flac=($(find stock/wav48_silence_trimmed -iname 'p2[0-4][0-9]*mic2.flac'))
unset IFS


n_flac=${#list_flac[@]}
echo $n_flac
n_comp=0.0

#n_flac=$(echo "$list_flac" | wc -l)

total_time=0

for item in ${list_flac[@]}
do
  let "n_comp++"


  pc=$(echo "$n_comp/$n_flac" | bc -l)
  v2=${pc:0:6}
  v3=$(echo "100*$v2" | bc -l | awk '{printf "%f", $0}' )
  v4=${v3::-3}
  item_length=$(soxi -D $item)

  clear && printf '\e[3J'

  echo "$v4 : $item"


  echo "len : $item_length"

  #total_time=$(echo "$total_time + $item_length" | bc -l)
  #echo $total_time


  percentBar $v4 $COLUMNS bar1
  echo "$bar1"

  typeset -F SECONDS=0

  #continue
  #

  #NOTE: TIMELINE:
  # 1 second: turn on mouse_logger, let values settle 
  #   | timestamp recorded by program, will be used later
  #   | PUT DATA IN vctk/gen/csv/[path]
  #   | [path] is the same as $item path
  # $item_length seconds: play sound 
  #   | RECORD TIMESTAMP
  #   | this timestamp will be used later along with the length of the file to cut off the CSV file automatically
  #   | this can be done with: flac -c -d $item | aplay, test on the speaker
  # 1 second: turn off mouse_logger, then wait 1s to settle
  #
  # repeat for all files
  #
  # TODO: do we need microphone, if so, we can just arecord [device] [path] with same structure as CSV directories
  #
  cd ..

  out_csv=$(echo $item | sed 's/^stock/gen\/csv/' | sed 's/\.flac$/.csv/')
  mkdir -p $(dirname "vctk/$out_csv")

  # extra 0.25ms is roughly the startup delay for the flac player
  timeout_length=$(echo "$item_length + 2.00025" | bc -l)

  
  trap sigint_handler SIGINT
  timeout -s INT $timeout_length sudo ./bin/mouse_logger "vctk/$out_csv" & pid_log=$(echo $!)
  #SECONDS=0


 

  sleep 1
  
  #echo $SECONDS

  # TODO: find the device ID of the speaker when plugged int
  flac -c -d -s "vctk/$item" | aplay -q & pid_flac=$(echo $!)

  #echo $SECONDS

  wait $pid_flac
  wait $pid_log

  cd vctk
done

#echo "total_time: $total_time seconds"

