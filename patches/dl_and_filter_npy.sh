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
country='us'

for tid in ${tar_ids[@]}; do

  echo "downloading $tid"
  ./azcopy cp "https://lilablobssc.blob.core.windows.net/geolifeclef-2020/patches_$country"_"$tid.tar.gz" "patches_$country"_"$tid.tar.gz"
  tar xvf "patches_$country"_"$tid.tar.gz"
  rm "patches_$country"_"$tid.tar.gz"

  echo "cleaning $tid"
  search_dir="./patches_$country"_"$tid"
  cd "$search_dir"

  # TODO: parse and keep the altitude files too (probably easiest to just include them in the txt)
  for entry in */*/*.npy; do
    if grep -q "$entry" ../annotations_train_top10_us_files.txt; then
      echo "keep $entry"
    else
      echo "delete $entry"
      rm $entry
    fi
  done

  cd ..

done
