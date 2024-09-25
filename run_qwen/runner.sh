#!/bin/bash

# Array of values
values=(0.5 1.5 3 7)

# Loop over the values and run the command with corresponding config files
for x in "${values[@]}"
do
    config_file="config$x.yml"
    echo "Running watermark-benchmark-run with $config_file"
    watermark-benchmark-run "$config_file"
done