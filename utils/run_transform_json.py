"""
This script takes an input text file and returns a json file.
Text file should contain an entry at each line.

Usage:
  >>  python run_transform_json.py
         --input clean_data.txt
         --output output.json

Args:
   input    : str, the path of the input text file
   output   : str, the path of the output file
   date     : str, optional, the date related to the dump file
       It is used to generate key template for the dict object
   delimiter: str, optinal, the delimiter in the text file
       It splits the integer ids from text content
   onlyfirst: bool, optional, set True if only first paragraphs will be collected
   random   : bool, optinal,set True if the key is generated randomly
"""

import argparse
from datetime import date
import json
import logging
from os.path import exists
from tqdm import tqdm
import unidecode

import preprocess_utils as prep, constants as cnt

logging.basicConfig(filename='log/data.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)


def generate_key_template(random_flag=False):
    """
    generates a template for the key of the dictionary
    :param random_flag: if set True random key generator is used else a defined template is used.
    returns:
       str, the key template
    """
    if random_flag:
        raise NotImplementedError("Random key generator is not implemented!")
    else:
        key_template = cnt.KEYTMP
    return key_template


def parse_args():
    """ Parse the arguments and returns the object """
    parser = argparse.ArgumentParser(description="The script to generate json from text file")
    parser.add_argument("--input",
                        help="Input file as text",
                        required=True)
    parser.add_argument("--output",
                        help="The path of the output folder",
                        required=True)
    parser.add_argument("--date",
                        help="The date of the data dump, used for the key",
                        default=None)
    parser.add_argument("--delimiter",
                        help="Delimiter used in the text to separate the content from index",
                        default="#")
    parser.add_argument("--onlyfirst",
                        help="If only first paragraphs will be collected",
                        default=True)
    parser.add_argument("--random",
                        help="Set True if keys are generated randomly",
                        action="store_true")
    args = parser.parse_args()
    return args


def function_return_dict(input_file, date_dt, key_template, take_first=True, delimiter="#"):
    """
    Reads a text file and returns dict object that contains the values.
    If the given file is json, it needs to store a list of dict object
    If the given file is csv, it needs to store a list of object.
    Args:
        :param input_file: str, the path of the input file
        :param date_dt: str, the date related to data dump
        :param key_template: str, the template used for the examples
        :param take_first: bool, set True if only first paragraph will be taken
        :param delimiter: str, the delimiter used to split
                 the integer id from the content
        :return:
            : dict, that contains the text data
    """

    with open(input_file) as fp:
        if input_file.endswith(cnt.JSON_EXT):
            data_dict = json.load(fp)
        else:
            data = fp.readlines()
            # create a dict from the list of text
            data_dict = {i: data[i].split(delimiter) for i in range(len(data))}

    # from default dict create structured dict
    prep_data_dict = {}
    if isinstance(data_dict, dict):
        for k, v in data_dict.items():
            if v != [cnt.NEWLINE]:
                key = key_template.format(date=date_dt,
                                          tag=prep.normalize_text(v[1]),
                                          tagId=v[0])
                if key in prep_data_dict:
                    # if the key is already exists
                    raise ValueError("Key %s is already in the dictionary" % key)
                else:
                    # assuming the order of the sample as
                    # Number, Name, DataField, table, detailed table, first paragraph, summary
                    if not (cnt.ERROR_MESSAGE in v[6]):
                        if prep.has_only_language_letters(v[6]):
                            prep_data_dict[key] = {"number": v[0], "name": v[1], "datafield": v[2],
                                                   "table": v[3], "detailed": v[4],
                                                   "paragraph": v[5], "summary": v[6]}
    elif isinstance(data_dict, list):
        for v in tqdm(data_dict):
            if v != [cnt.NEWLINE]:
                key = key_template.format(date=date_dt,
                                          tag=unidecode.unidecode(v['title'].lower()),
                                          tagId=v['id']).\
                    replace(cnt.SPACE, cnt.EMPTY).\
                    replace(cnt.FileSEP, cnt.EMPTY)
                if key in prep_data_dict:
                    # if the key is already exists
                    raise ValueError("Key %s is already in the dictionary" % key)
                else:
                    if not (cnt.ERROR_MESSAGE in v["Article_Clean_Texts"]):
                        # Checks if the paragraph
                        if take_first is True:
                            if len(v['Article_Clean_Texts']['Paragraphs']) > 0:
                                paragraph = v['Article_Clean_Texts']['Paragraphs'][0]
                        else:
                            paragraph = v['Article_Clean_Texts']['Paragraphs']
                        # Check if the paragraph contains characters
                        # that don't belong to the language
                        if paragraph:
                            if prep.has_only_language_letters(paragraph[0]):
                                prep_data_dict[key] = {"number": v['id'],
                                                       "name": v['title'],
                                                       "datafield": v['InfoBox_type'],
                                                       "table": v['InfoBox_Clean_jsonText'],
                                                       "paragraph": paragraph}

    return prep_data_dict


def save_json(data_dict, output_path):
    """
    saves the given dict to the output path
    :param data_dict: dict, that contains the examples
    :param output_path: str, where to save the file
    :return:
       None
    """
    if exists(output_path):
        raise FileExistsError("%s exists" % output_path)
    with open(output_path, "w+") as fp:
        json.dump(data_dict, fp, indent=4)
    logger.info("The dict was saved into %s, contains %d samples" % (output_path, len(data_dict)))


def main():
    """
    Runs all the process at once.
    Open the text file, generates dict object, saves into json file
    :return:
        None
    """
    # Get the inputs
    inputs = parse_args()

    input_file = inputs.input
    logger.info("Starting the script using input file %s" % input_file)
    output_file = inputs.output

    if inputs.date is None:
        # if no date is given, take the today
        date_today = date.today().strftime("%d%m%Y")
    else:
        date_today = inputs.date
    # generates a dict object
    data_dict = function_return_dict(input_file,
                                     date_today,
                                     key_template=generate_key_template(inputs.random),
                                     take_first=inputs.onlyfirst,
                                     delimiter=inputs.delimiter)
    # save the dict object
    save_json(data_dict, output_file)
    logger.info("Finished the process.")
    return


if __name__ == "__main__":
    main()
