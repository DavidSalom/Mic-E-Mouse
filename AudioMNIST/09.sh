#!/bin/bash

# Loop through the file indices 0 to 9
for index in {0..9}
do
    # Run the ffmpeg command with the current index
    ffmpeg -i "data/01/${index}_01_0.wav" -filter:a "volume=50dB" -f matroska - | ffplay - -autoexit -nodisp

    # Sleep for 1 second
    sleep 1
done

