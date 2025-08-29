#!/bin/bash

# Directory to search in - you can modify this path
SEARCH_DIR="./benchmark/benchmark_data/pytorch_profiler"

# First list all files that will be removed (dry run)
echo "The following files will be removed:"
find "$SEARCH_DIR" -type f \( -name "*rank_1*" -o -name "*rank_2*" -o -name "*rank_3*" -o -name "*rank_4*" -o -name "*rank_5*" -o -name "*rank_6*" -o -name "*rank_7*" -o -name "*rank_8*" -o -name "xfuser_flux_trace_steps_*" \) -print

# Ask for confirmation
read -p "Are you sure you want to delete these files? (y/n) " -n 1 -r
echo    # Move to a new line

if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Actually remove the files
    find "$SEARCH_DIR" -type f \( -name "*rank_1*" -o -name "*rank_2*" -o -name "*rank_3*" -o -name "*rank_4*" -o -name "*rank_5*" -o -name "*rank_6*" -o -name "*rank_7*" -o -name "*rank_8*" -o -name "xfuser_flux_trace_steps_*" \) -delete
    echo "Files have been removed."
else
    echo "Operation cancelled."
fi
