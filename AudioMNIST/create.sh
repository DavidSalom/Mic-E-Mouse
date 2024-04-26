#!/bin/zsh


function sigterm_handler() {
  # happens at 7am until 8pm = 46800 seconds
  echo "waiting for interval -- sleeping from 7am to 8pm - create"
  current_epoch=$(date +%s)
  target_epoch=$(date -d 'today 20:00' +%s)
  #target_epoch=$(date  "+%s" -d "1 second")
  sleep_seconds=$(( $target_epoch - $current_epoch ))


  if [[ $sleep_seconds -ge 0 ]]
  then
    echo "sleeping for $sleep_seconds seconds"
    sleep $sleep_seconds | pv -t
  fi
}

trap sigterm_handler SIGTERM
echo "$$" > c_pid


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

boost_level=$3
if [[ -z "$boost_level" ]]; then
  boost_level="50";
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
  echo "01" > c_spkr
fi
c_spkr=$(cat < ./c_spkr)
list_dir=($c_spkr)
list_dir+=($(ls data | sort | sed "0,/^$c_spkr$/d"))

IFS=$'\n'
for item in ${list_dir[@]}
do
  list_wavs+=($(find data/$item -iname "*.wav"))
  #echo ${#list_flac[@]}
  #echo $item
done
unset IFS


n_wavs=${#list_wavs[@]}
printf 'NUM WAVS: %s\n' "${list_wavs[@]}"
n_all_wavs=$(find data -iname '*.wav'  | wc -l)
echo $n_wavs
n_comp=$(echo "$n_all_wavs - $n_wavs" | bc)

#n_flac=$(echo "$list_wavs" | wc -l)


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

      if [[ ! "-f ../$item" || "$i_ext" -ne "wav" ]]
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

  n_wavs=$#q_pf
  n_all_wavs=$#q_pf
  n_comp=0


  list_wavs=("${q_pf[@]}")

  
  

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
sil_name="AudioMNIST/noise/noise_$sil_date.csv"
trap sigint_handler SIGINT
timeout -s INT $silence_length sudo ./bin/mouse_logger "$sil_name" $mouse_type &
pid_log=$(echo $!)

cd AudioMNIST

seconds=$silence_length
start="$(($(date +%s) + $seconds))"
while [ "$start" -ge `date +%s` ]; do
    time="$(( $start - `date +%s` ))"
    sudo true
    printf '%s\r' "$(date -u -d "@$time" +%M:%S)"
done

wait $pid_log

# main recording section

for item in ${list_wavs[@]}
do
  let "n_comp++"

  sudo true


  echo $item



  pc=$(echo "$n_comp/$n_all_wavs" | bc -l)
  v2=${pc:0:6}
  v3=$(echo "100*$v2" | bc -l | awk '{printf "%f", $0}' )
  v4=${v3::-3}
  item_length=$(soxi -D $item)

  clear && printf '\e[3J'

  echo "$v4 : $item"
  base_file=$(basename $item)
  base_file=${base_file:2:2}
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
  #   | PUT DATA IN AudioMNIST/gen/csv/[path]
  #   | [path] is the same as $item path
  # $item_length seconds: play sound 
  #   | RECORD TIMESTAMP
  #   | this timestamp will be used later along with the length of the file to cut off the CSV file automatically
  #   | this can be done with: flac -c -d $item | aplay, test on the speaker
  # 1 second: turn off mouse_logger, then wait 1s to settle
  #
  # repeat for all files
  #
  cd ..

  out_csv=$(echo $item | sed 's/^data/gen\/csv/' | sed 's/\.wav$/.csv/')
  mkdir -p $(dirname "AudioMNIST/$out_csv")

  # extra 0.25ms is roughly the startup delay for the flac player


  
  trap sigint_handler SIGINT
  timeout -s INT $timeout_length sudo ./bin/mouse_logger "AudioMNIST/$out_csv" $mouse_type & pid_log=$(echo $!)
  #SECONDS=0


 

  #sleep 1
  
  #echo $SECONDS

  ffmpeg -i "AudioMNIST/$item" -filter:a "volume=""$boost_level""dB" -f matroska - | ffplay - -autoexit -nodisp
  wait $pid_log


  cd AudioMNIST
done
