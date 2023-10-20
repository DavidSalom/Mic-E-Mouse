#!/bin/zsh

#if [[ ! $(sudo echo 0) ]]; then exit; fi

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

  #continue


  #// TODO: actually implement recording
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
 

  # TODO: ENABLE MOUSE_LOGGER WITH PATH SUPPORT (via. argv)
  #
  cd ..

  out_csv=$(echo $item | sed 's/^stock/gen\/csv/')
  mkdir -p $(dirname "vctk/$out_csv")

  timeout_length=$(echo "$item_length + 2.0" | bc -l)

  #TODO: mouse_logger does not write to file until ANOTHER mouse interrupt occurs, needs to be fixed
  #timeout -s INT $timeout_length sudo ./bin/mouse_logger "vctk/$out_csv" & pid_log=$(echo $!)


 

  sleep 1

  # TODO: find the device ID of the speaker when plugged int
  #flac -c -d "vctk/$item" | aplay & pid_flac=$(echo $!)

  #wait $pid_flac
  #wait $log_flac

  



  cd vctk

  #echo $n_comp
done

#echo "total_time: $total_time seconds"

