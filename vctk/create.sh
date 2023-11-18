#!/bin/zsh


#if [[ ! $(sudo echo 0) ]]; then exit; fi

pid_log=0

# NOTE: 
# [0]: RAZER 8KHz
# [1]: G502
# if mouse_type is not provided, default to RAZER 8KHz, we can provide mouse_type using the first argument
mouse_type=$1
if [[ -z "$mouse_type" ]]; then
  mouse_type=0;
fi

path_file=$2
if [[ -z "$path_file" ]]; then
  path_file="NONE";
fi

echo $path_file

export TERMINFO=/usr/share/terminfo

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

if [[ ! -s ./c_spkr || ! -f ./c_spkr ]]
then
  echo "log.txt" > c_spkr
fi

c_spkr=$(cat < ./c_spkr)
echo $c_spkr
list_dir=($c_spkr)
list_dir+=($(ls stock/wav48_silence_trimmed | sort | sed "0,/^$c_spkr$/d"))

IFS=$'\n'
for item in ${list_dir[@]}
do
  list_flac+=($(find stock/wav48_silence_trimmed/$item -iname "$item*mic2.flac"))
  #echo ${#list_flac[@]}
  #echo $item
done
unset IFS


n_flac=${#list_flac[@]}

n_all_flac=$(find stock/wav48_silence_trimmed -iname '*mic2.flac'  | wc -l)
echo $n_flac
n_comp=$(echo "$n_all_flac - $n_flac" | bc)

#n_flac=$(echo "$list_flac" | wc -l)


if [[ $path_file != "NONE" ]]
then

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


  list_flac=("${q_pf[@]}")

  
  

  #sleep 100
fi



total_time=0
unint_time=0
max_unint_time=3600


#l=""
#for item in ${list_flac[@]}
#do
  #base_file=$(basename $item)
  #base_file=${base_file:0:4}

  #if [[ "$l" != "$base_file" ]]
  #then
    #echo "$base_file"
    #l=$base_file
  #fi

#done

# recording 4 min of silence
silence_length=1

echo "recording $silence_length seconds of silence:"

if [[ ! $(sudo echo 0) ]]; then exit; fi



cd ..
sil_date=$(date +%Y-%m-%d_%H-%M-%S)
sil_name="vctk/noise/noise_$sil_date.csv"
trap sigint_handler SIGINT
timeout -s INT $silence_length sudo ./bin/mouse_logger "$sil_name" $mouse_type &
pid_log=$(echo $!)

cd vctk

seconds=$silence_length
start="$(($(date +%s) + $seconds))"
while [ "$start" -ge `date +%s` ]; do
    time="$(( $start - `date +%s` ))"
    sudo true
    printf '%s\r' "$(date -u -d "@$time" +%M:%S)"
done

wait $pid_log

# main recording section

for item in ${list_flac[@]}
do
  let "n_comp++"

  sudo true


  echo $item



  pc=$(echo "$n_comp/$n_all_flac" | bc -l)
  v2=${pc:0:6}
  v3=$(echo "100*$v2" | bc -l | awk '{printf "%f", $0}' )
  v4=${v3::-3}
  item_length=$(soxi -D $item)

  clear && printf '\e[3J'

  echo "$v4 : $item"
  base_file=$(basename $item)
  base_file=${base_file:0:4}
  echo "dir : $base_file"
  echo "$base_file" > c_spkr


  echo "len : $item_length"

  timeout_length=$(echo "$item_length + 2.00025" | bc -l)
  
  unint_time=$(echo "$unint_time + $timeout_length" | bc -l)

  echo "uninterrupted time (max $max_unint_time): $unint_time"

  #total_time=$(echo "$total_time + $item_length" | bc -l)
  #echo $total_time


  percentBar $v4 $COLUMNS bar1
  echo "$bar1"

  typeset -F SECONDS=0


  if ((unint_time>max_unint_time))
  then
    echo "$max_unint_time seconds have passed, taking a 30 minute break"

    unint_time=0

    seconds=300
    start="$(($(date +%s) + $seconds))"
    while [ "$start" -ge `date +%s` ]; do
        time="$(( $start - `date +%s` ))"
        sudo true
        printf '%s\r' "$(date -u -d "@$time" +%M:%S)"
    done
  fi
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


  
  trap sigint_handler SIGINT
  timeout -s INT $timeout_length sudo ./bin/mouse_logger "vctk/$out_csv" $mouse_type & pid_log=$(echo $!)
  #SECONDS=0


 

  sleep 1
  
  #echo $SECONDS

  # TODO: find the device ID of the speaker when plugged int
  flac -c -d -s "vctk/$item" | aplay -q & pid_flac=$(echo $!)

  #echo $SECONDS

  wait $pid_flac
  wait $pid_log

  #trap - SIGINT

  cd vctk
done

#echo "total_time: $total_time seconds"

