#!/bin/zsh
#

IFS=$'\n'
list_dir=($(ls stock/wav48_silence_trimmed))
unset IFS

total_time=0

for n in ${list_dir[@]}
do
  #echo $n

  IFS=$'\n'
  list_flac=($(find stock/wav48_silence_trimmed/$n -iname 'p[0-9][0-9][0-9]*mic2.flac'))
  unset IFS

  n_time=0

  n_flac=${#list_flac[@]}
  #echo $n_flac

  for item in ${list_flac[@]}
  do
    let "n_comp++"



    item_length=$(soxi -D $item)

    n_time=$(echo "$n_time + $item_length + 0.02" | bc -l)

  done
  total_time=$(echo "$n_time + $total_time" | bc -l)
  echo "$n: $n_time \t (total: $total_time)"
done
exit

