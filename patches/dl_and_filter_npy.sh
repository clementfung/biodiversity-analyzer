#!/bin/bash
#!/usr/bin/env bash
set -euo pipefail

### BE CAREFUL. This script downloads, reads and deletes files in a loop.

## A fairly messy script that will:
# (i) download all 20 tar.gz files, one at a time
# (ii) untar them 
# (iii) search the new directory for any file not listed in the filtered txt file
# (iv) and deletes them if not found

## Requirements: 
# (1) azcopy is downloaded and in this (./patches) directory
# (2) the previous output of json_parser.py (which contains the filenames of the filtered json records) is in this (./patches) directory

# Redefine the array as needed
tar_ids=('01' '02' '03' '04' '05' '06' '07' '08' '09' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20')
#tar_ids=('06' '07' '08' '09')

for tid in ${tar_ids[@]}; do

  echo "downloading $tid"  
  ./azcopy cp "https://lilablobssc.blob.core.windows.net/geolifeclef-2020/patches_fr_$tid.tar.gz" "patches_fr_$tid.tar.gz"
  tar xvf "patches_fr_$tid.tar.gz"
  rm "patches_fr_$tid.tar.gz"

  echo "cleaning $tid"  
  search_dir="./patches_fr_$tid"
  cd "$search_dir"

  # TODO: parse and keep the altitude files too (probably easiest to just include them in the txt)
  for entry in */*/*.npy; do
    if grep -q "$entry" ../annotations_train_filtered_fr_files.txt; then
      echo "keep $entry" 
    else
      echo "delete $entry"
      rm $entry
    fi
  done

  cd ..

done
