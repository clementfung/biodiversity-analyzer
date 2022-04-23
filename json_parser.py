import os, json, sys
import pdb
import numpy as np

def count_categories(filename):

    with open(filename, "r") as f:        
        x = json.load(f)

    num_categories = x['categories'][-1]['id'] + 1
    counts = np.zeros(num_categories)

    for i in range(len(x['annotations'])):
        this_category = x['annotations'][i]['category_id']
        counts[this_category] += 1

    top_k = np.argsort(counts)[-10:]
    total_sum = 0

    for i in top_k:
        print(f'Most common: {counts[i]} {x["categories"][i]["gbif_name"]}')
        total_sum += counts[i]

    print(f'A total of {total_sum} records in top 10')
    return top_k

def parse_annotations(filename, output_tag, filtering=False, save_txt=True):

    total_records_fr = []
    total_filenames_fr = []
    total_filenames_alti_fr = []

    total_records_us = []
    total_filenames_us = []
    total_filenames_alti_us = []

    with open(filename, "r") as f:
        x = json.load(f)

    print(len(x['images']))
    print(len(x['annotations']))

    top_k_categories = count_categories(filename)

    for i in range(len(x['images'])):
        image_id = x['images'][i]['id']
        tag_id = x['annotations'][i]['image_id']

        if image_id != tag_id:
            print(f'SOMETHING WENT WRONG AT RECORD {i}')
            return

        category_id = x['annotations'][i]['category_id']

        # cindy: top 10? 20?
        # Use only the top 20 categories
        if filtering and category_id not in top_k_categories:
            continue
            
        new_record = dict()
        for key in x['images'][i].keys():

            new_record[key] = x['images'][i][key]
            # cindy: following line should be out of for loop?
            new_record['category'] = category_id

        country = x['images'][i]["country"]

        if country == "fr":
            total_filenames_fr.append(x['images'][i]['file_name'])
            total_filenames_alti_fr.append(x['images'][i]['file_name_alti'])
            total_records_fr.append(new_record)
        elif country == "us":
            total_filenames_us.append(x['images'][i]['file_name'])
            total_filenames_alti_us.append(x['images'][i]['file_name_alti'])
            total_records_us.append(new_record)
        else:
            print(f'Unknown country: {country}')

    total_records = total_records_us + total_records_fr
    total_filenames = total_filenames_us + total_filenames_fr

    print(f'Final output length (fr): {len(total_records_fr)}')
    print(f'Final output length (us): {len(total_records_us)}')
    print(f'Final output length: {len(total_records)}')

    if save_txt:

        if filtering:
            output_tag = f'{output_tag}_top10'

        #####################
        # Save france files
        #####################
        with open(f'{output_tag}_fr_parsed.json', "w") as output:
            json.dump(total_records_fr, output)

        with open(f'{output_tag}_fr_files_rgb.txt', "w") as output:
            for file in sorted(total_filenames_fr):
                output.write(f'{file}\n')

        with open(f'{output_tag}_fr_files_alti.txt', "w") as output:
            for file in sorted(total_filenames_alti_fr):
                output.write(f'{file}\n')

        # Write both to same file
        with open(f'{output_tag}_fr_files.txt', "w") as output:
            for file in sorted(total_filenames_fr):
                output.write(f'{file}\n')

            for file in sorted(total_filenames_alti_fr):
                output.write(f'{file}\n')

        #####################
        # Save USA files
        #####################
        with open(f'{output_tag}_us_parsed.json', "w") as output:
            json.dump(total_records_us, output)

        with open(f'{output_tag}_us_files_rgb.txt', "w") as output:
            for file in sorted(total_filenames_us):
                output.write(f'{file}\n')

        with open(f'{output_tag}_us_files_alti.txt', "w") as output:
            for file in sorted(total_filenames_alti_us):
                output.write(f'{file}\n')

        with open(f'{output_tag}_us_files.txt', "w") as output:
            for file in sorted(total_filenames_us):
                output.write(f'{file}\n')

            for file in sorted(total_filenames_alti_us):
                output.write(f'{file}\n')

        # Save total files
        with open(f'{output_tag}_parsed.json', "w") as output:
            json.dump(total_records, output)

        with open(f'{output_tag}_files.txt', "w") as output:
            for file in sorted(total_filenames):
                output.write(f'{file}\n')

if __name__ == '__main__':
    
    filename = sys.argv[1]

    # Remove the '.json from the filename'
    output_tag = filename[:-5]

    parse_annotations(filename, output_tag, filtering=True)
