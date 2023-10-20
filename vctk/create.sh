#!/bin/zsh


n_flac=$(find stock/wav48_silence_trimmed -iname 'p2[0-4][0-9]*.flac' | wc -l)


echo $n_flac
