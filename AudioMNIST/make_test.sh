#!/bin/zsh

zmodload zsh/mapfile
FNAME=path.txt
FLINES=( "${(f)mapfile[$FNAME]}" )
for ITEM in $FLINES
do
    f=${ITEM:5}
    echo $f
    mkdir -p $(dirname "data/"$f)
    cp "data_all/"$f "data/"$f
done
