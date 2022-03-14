import os, json, sys
import pdb

def parse_annotations(filename, output_tag, filtering=False):

    total_records = []
    total_filenames = []

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

            # Try using only the first 10 categories
            if filtering and category_id > 20:
                continue

            new_record = dict()
            for key in x['images'][i].keys():

                new_record[key] = x['images'][i][key]
                new_record['category'] = category_id

            total_filenames.append(x['images'][i]['file_name'])
            total_records.append(new_record)

    print(f'Final output length: {len(total_records)}')

    if filtering:
        output_tag = f'{output_tag}_filtered'

    with open(f'{output_tag}_parsed.json', "w") as output:
        json.dump(total_records, output)

    with open(f'{output_tag}_files.txt', "w") as output:
        for file in sorted(total_filenames):
            output.write(f'{file}\n')

if __name__ == '__main__':
    
    filename = sys.argv[1]

    # Remove the '.json from the filename'
    output_tag = filename[:-5]

    parse_annotations(filename, output_tag, filtering=False)
