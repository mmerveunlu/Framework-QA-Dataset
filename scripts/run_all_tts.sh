# This script contains example run to generate TTS data for a given textual data
# For more detail, please look the file module_tts/README.md
# Before running the scripts, make sure to change the parameters accordingly.

# First step is to transform the json data into clean json format
python utils/run_transform_json.py --input data/example_subset_tr_wiki.json --output data/example_subset_tr_wiki_clean.json

# Second step is filtering
python utils/run_filter_data.py --input data/example_subset_tr_wiki_clean.json --output data/example_subset_tr_wiki_filtered.json --min_word 4 --max_word 1000 --non_type

# Third step is collecting TTS
python module_tts/run_tts.py --input data/example_subset_tr_wiki_filtered.json --keyname paragraph --output data/example_tr_wiki/ --language tr --split window --dtsplit