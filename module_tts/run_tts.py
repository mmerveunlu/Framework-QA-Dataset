"""
This script runs model_tts on the given input file.

Usage:
  >> python run_tts.py
      --input data/cleaned_data.json
      --keyname summary
      --output data/example_data/
      --language tr

Args:
    input    : str, the path of the json file to be read
    keyname  : str, the key in the json file to be transformed into speech
    output   : str, the path of the output folder
    language : str, the abbreviation for the language: tr, eng etc.
    split    : str, the mode how to split the text
       If speech data will be word based, then split is "word"
       If speech data will be sentence based, then split is "sentence"
       If split is not given, default is using the given text as it is.
    onlytext: bool, if only text will be generated, no pass for tts
    dtsplit : bool, if given the script further creates train,test splits from the dataset

Output:
  In the output folder, the model saves all the speech data with the name as [subject].[name].[ID]
  If splitting mode is used, then there are several speech data generated from text.
  All the files are saved as : [subject].[name].[ID].[XXX] where XXX indicates the index of the split.
"""

import os
import logging
import argparse

import google_tts_wrap as tts

FORMAT = "%(asctime)s - %(name)s - %(levelname)s -[%(filename)s:%(lineno)s - %(funcName)20s() ]- %(message)s"
logging.basicConfig(filename='../log/data.log',
                    level=logging.INFO,
                    format=FORMAT,
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger(__name__)


def parse_args():
    """ Argument parser"""
    parser = argparse.ArgumentParser(description='Generate TTS from given file')
    parser.add_argument('--input',
                        help='input file name',
                        required=True)
    parser.add_argument('--keyname',
                        help='Key name to get text data',
                        default="subject",
                        required=True)
    parser.add_argument('--output',
                        help='Output folder names',
                        required=True)
    parser.add_argument('--language',
                        help='Language abbreviation for speech',
                        default="tr")
    parser.add_argument('--split',
                        help="How to split the text, can be: word, sentence, None")
    parser.add_argument('--onlytext',
                        help='Set True if only text data is generated',
                        action='store_true')
    parser.add_argument('--dtsplit',
                        help='Set True if train/test split is generated',
                        action='store_true')
    args = parser.parse_args()
    return args


def main():
    """
    Runs all the process at once.
    Open the json file, reads text samples, generates speech and saves them
    :return:
        None
    """
    # Get the inputs
    args = parse_args()
    input_file = args.input
    output_folder = args.output
    logger.info("Starting the script using input file %s" % input_file)

    # Check the folders
    if not os.path.isfile(input_file):
        raise FileNotFoundError("The specified path doesn't exist: %s" % input_file)

    if not os.path.isdir(output_folder):
        logger.info('Creating output folder %s' % output_folder)
        os.mkdir(args.output)

    if not args.onlytext:
        tts.generate_tts_from_file(input_file,
                                   args.keyname,
                                   output_folder,
                                   args.language,
                                   args.split)
        logger.info("Generating TTS is done!")
    elif args.onlytext:
        tts.generate_text_file(input_file,
                               args.keyname,
                               output_folder,
                               args.language,
                               args.split)
        logger.info("Generating text is done!")
    if args.dtsplit:
        tar_name = output_folder + ".tar"
        tts.generate_train_test_splits(output_folder, tar_name, args.language)
        logger.info("Train/Test Split generation is done!")


if __name__ == "__main__":
    main()
