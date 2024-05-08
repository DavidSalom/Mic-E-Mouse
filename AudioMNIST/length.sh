#!/bin/zsh
#

IFS=$'\n'
list_dir=($(ls data))
unset IFS

total_time=0

for n in ${list_dir[@]}
do
  #echo $n

  IFS=$'\n'
  list_wav=($(find data/$n -iname '*.wav'))
  unset IFS

  n_time=0

  n_wav=${#list_wav[@]}
  #echo $n_wav

  for item in ${list_wav[@]}
  do
    let "n_comp++"



    item_length=$(soxi -D $item)

    n_time=$(echo "$n_time + $item_length + 0.2" | bc -l)

  done
  total_time=$(echo "$n_time + $total_time" | bc -l)
  echo "$n: $n_time \t (total: $total_time)"
done
exit

