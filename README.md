# biodiversity-analyzer
Course project for CMU 17-737 AI for Social Good.
Data is collected from [GeoLifeCLEF 2020](https://lila.science/datasets/geolifeclef-2020/)

## Code setup (use virtualenv)

Set up a local virtual environment to install and manage package versions.

1. `virtualenv -p python3 venv`  
2. `source venv/bin/activate`  
3. `pip3 install -r requirements.txt`  

## Useful scripts

### json_parser.py

Given an input of annotations (one of `annotations_train.json` or `annotations_val.json`), this script will read and re-format the JSON object accordingly.  
There are two outputs: (1) a reformatted JSON which includes the category_id (label) with each record, and (2) a list of all included filenames, sorted.

Optionally, this script can also filter out all category_id's above a certain value.  
i.e. It's useful to restrict the eventual ML model to the top 20 classes for easier training and eval.

Example usage: `python3 json_parser.py annotations_train.json` (see the file for how the filtering works)

### process_patches.py

> Mostly still an example for now

Given a set of input `.npy` files, this script will separate out the data format into n-by-d formatting matrices, ready to be used for machine learning.

### train_model.py

> Mostly still an example for now

Given a set of input processed `.npy` files (the output of `process_patches.py`), this script will perform a very basic CNN training.
