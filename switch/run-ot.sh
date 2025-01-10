#!/bin/bash

declare -a i_values=(0 1 2 3 4 5 6 7)


# Loop over each combination of 'k' and 's' values
for i in "${i_values[@]}"
do

    mkdir -p extract_saved_8-${i}
    
    python switch-OT.py --q "$i"

done
