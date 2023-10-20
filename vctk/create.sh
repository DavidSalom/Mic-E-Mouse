#!/bin/zsh
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
list_flac=($(find stock/wav48_silence_trimmed -iname 'p2[0-4][0-9]*.flac'))
unset IFS


n_flac=${#list_flac[@]}
echo $n_flac
n_comp=0.0


n_flac=$(find stock/wav48_silence_trimmed -iname 'p2[0-3][0-9]*.flac' | wc -l)
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

  total_time=$(echo "$total_time + $item_length" | bc -l)
  echo $total_time


  percentBar $v4 $COLUMNS bar1
  echo "$bar1"
  

  #echo $n_comp
done

echo "total_time: $total_time seconds"

