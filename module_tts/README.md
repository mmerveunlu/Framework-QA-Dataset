# How to generate TTS from text file

This folder contains the audio generation step given text file. 
It is a 3-step process to collect TTS data for textual data.

We can run all the scripts together or separately. 
To run the all scripts at once, please follow section __Running all at once__. 
To run the scripts separately, please follow section __Running separately__.

## Running all at once 

Please make sure that you change the necessary parameters in the script file: 

 > sh scripts/run_all_tts.sh

## Running separately

1. The script utils/run_transform_json.py runs this step.
   * removes the examples that contain non-language characters 
   * removes the examples that don't contain any paragraphs 
   * removes the examples that contain error in paragraphs

 > python utils/run_transform_json.py \ 
    --input data/example_subset_tr_wiki.json \
    --output data/example_subset_tr_wiki_clean.json
   
Clean file contains a dictionary where keys are unique ids assigned for each example,
and values are dicts with {"number","name","datafield","table","detailed","paragraph","summary"}

2. (Optional) After generating json dump, we can filter the data with:
   * by the number of the word: removes the examples whose word numbers are not in the given range
   * by the existence of data field: removes the examples without type information
   * by the data field: Choosing the most frequent data fields only 

> python utils/run_filter_data.py \
    --input data/example_subset_tr_wiki_clean.json \
    --output data/example_subset_tr_wiki_filtered.json \
    --min_word 4 \
    --max_word 1000  \
    --non_type

3. We need to run tts script to collect audio files for each example 

> python module_tts/run_tts.py \
   --input data/example_subset_tr_wiki_filtered.json \
   --keyname paragraph \
   --output data/example_tr_wiki/ \
   --language tr \
   --split window \
   --dtsplit

The script can take a while to finish, depending on the number of request.
In the end it generates, a tar file that contains audio file under [lang]/clips/ and train/test split sentences in the files train.csv and test.csv.

If any steps, tts fails. Then you need to run the next command after tts completion. Since TTS failed, text data is not saved for whole data. 

> python module_tts/run_tts.py \
   --input data/example_subset_tr_wiki_filtered.json \
   --keyname paragraph \
   --output data/example_tr_wiki/ \
   --language tr \
   --onlytext \
   --split window \
   --dtsplit

