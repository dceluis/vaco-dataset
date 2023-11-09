import glob
import json
import sys

# read glob from arguments or default to *.json
glob_pattern = sys.argv[1] if len(sys.argv) > 1 else "*.json"
read_files = glob.glob(glob_pattern)

output_list = []

for f in read_files:
    with open(f, "rb") as infile:
        output_list.append(json.load(infile))

all_items = []

for json_file in output_list:
   all_items.extend(json_file)

textfile_merged = open('import.json', 'w')

textfile_merged.write(json.dumps(all_items, indent=4, sort_keys=True))
