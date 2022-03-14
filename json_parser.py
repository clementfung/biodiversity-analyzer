import os, json, sys
import pdb

def parse_annotations(filename, output_tag, filtering=False):

    total_records_fr = []
    total_filenames_fr = []

    total_records_us = []
    total_filenames_us = []

    with open(filename, "r") as f:
        
        x  = json.load(f)

        print(len(x['images']))
        print(len(x['annotations']))

        for i in range(len(x['images'])):
            image_id = x['images'][i]['id']
            tag_id = x['annotations'][i]['image_id']

            if image_id != tag_id:
                print(f'SOMETHING WENT WRONG AT RECORD {i}')
                return

            category_id = x['annotations'][i]['category_id']

            # Try using only the first 20 categories
            if filtering and category_id > 20:
                continue
                
            new_record = dict()
            for key in x['images'][i].keys():

                new_record[key] = x['images'][i][key]
                new_record['category'] = category_id

            country = x['images'][i]["country"]

            if country == "fr":
                total_filenames_fr.append(x['images'][i]['file_name'])
                total_records_fr.append(new_record)
            elif country == "us":
                total_filenames_us.append(x['images'][i]['file_name'])
                total_records_us.append(new_record)
            else:
                print(f'Unknown country: {country}')

    print(f'Final output length (fr): {len(total_records_fr)}')
    print(f'Final output length (us): {len(total_records_us)}')

    if filtering:
        output_tag = f'{output_tag}_filtered'

    # Save france files
    with open(f'{output_tag}_fr_parsed.json', "w") as output:
        json.dump(total_records_fr, output)

    with open(f'{output_tag}_fr_files.txt', "w") as output:
        for file in sorted(total_filenames_fr):
            output.write(f'{file}\n')

    # Save US files
    with open(f'{output_tag}_us_parsed.json', "w") as output:
        json.dump(total_records_us, output)

    with open(f'{output_tag}_us_files.txt', "w") as output:
        for file in sorted(total_filenames_us):
            output.write(f'{file}\n')

if __name__ == '__main__':
    
    filename = sys.argv[1]

    # Remove the '.json from the filename'
    output_tag = filename[:-5]

    parse_annotations(filename, output_tag, filtering=True)
