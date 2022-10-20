import csv
import json
import logging
import os
import re
from os.path import exists
import shutil
import sys
import tarfile
import time

from gtts import gTTS
from tqdm import tqdm
import pydub
from sklearn.model_selection import train_test_split

from utils import preprocess_utils as prep, constants as cnt

FORMAT = "%(asctime)s - %(name)s - %(levelname)s -[%(filename)s:%(lineno)s - %(funcName)20s() ]- %(message)s"
logging.basicConfig(filename='log/data.log',
                    level=logging.INFO,
                    format=FORMAT,
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
REPLACE_DICT = {",": "", "&": "_", "'": "_", "?": "_", "\"": "_", "*": "_", "__": "_", ";": "_"}


def create_tar_file(path, tar_name):
    """ Creates a tar file from given path """
    with tarfile.open(tar_name, "w") as t:
        t.add(path, arcname=os.path.basename(path))


def write_text_csv(text_dict, out_text_file):
    """
    Saves the dict file to csv file
    :param text_dict: dict, contains path and sentence of the audio
    :param out_text_file: string, the path of the out file
    :return: none
    """
    with open(out_text_file, 'w', newline='') as csvfile:
        fieldnames = ['path', 'sentence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key in text_dict:
            writer.writerow({'path': key, 'sentence': text_dict[key]})


class ErrorAudioFile(Exception): pass


class SkipTheExample(Exception): pass


class TTSError(Exception): pass


def test_audio_file(filepath):
    """ Test the created audio file
    If it can't open, then skip that example
    :param filepath: string, the path of the audio file
    :return int, If file opens 1, else -1
    """
    try:
        a = pydub.AudioSegment.from_mp3(filepath)
        return 1
    except ErrorAudioFile:
        logger.debug("Decoding of file %s fails, skipping the example. ", filepath)
        return -1


def save_tsv_file(filepath, content):
    """
    Saves the given content into filepath
    :param filepath: str, the path of the output
    :param content: dict, keys are path and values are sentences
    :return: None
    """
    with open(filepath, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['path', 'sentence'])
        for k, v in content.items():
            tsv_writer.writerow([k, v])
    logger.info("The sentences are saved into %s ", filepath)


def generate_train_test_splits(input_folder, output_folder, language, test_size=0.33):
    """
    Reads the folder and splits the sentences into train, test and valid.
    :param input_folder  : str, it contains clips and sentences
    :param output_folder : str, the path of output tar file
    :param language      : str, the abbreviation of the language
    :param test_size     : int, the rate of train/test split
    :return None
    """
    # create folder under data
    lang_folder = os.path.join(input_folder, language)
    if not os.path.exists(lang_folder):
        os.mkdir(lang_folder)

    # read the sentences
    sent_file = os.path.join(input_folder, "sentences.csv")
    with open(sent_file) as fp:
        lines = fp.readlines()
    data = {}
    f_keys = []
    for line in lines:
        key = line.split(",")[0]
        if not (key == "path"):
            if not (cnt.ERROR_MESSAGE in line):
                data[key] = ",".join(line.split(",")[1:])
                f_keys.append(".".join(key.split(".")[0:5]))

    # removing duplicate values
    f_keys = list(set(f_keys))
    # split the keys into train and test
    if test_size > 0:
        train_keys, test_keys = train_test_split(f_keys,
                                                 test_size=test_size,
                                                 random_state=42)
    else:
        train_keys = f_keys
        test_keys = []
    # now split the sentences
    train_sentences = {}
    test_sentences = {}
    for k, v in data.items():
        fk = ".".join(k.split(".")[0:5])
        if fk in train_keys:
            train_sentences[k] = v
        elif fk in test_keys:
            test_sentences[k] = v
        else:
            raise KeyError("Key %s is not found" % fk)

    logger.info("The number of train split by passages: %d, test split %d", len(train_keys), len(test_keys))
    logger.info("The number of train split by sentences: %d, test split %d", len(train_sentences), len(test_sentences))
    logger.info("The total number of sentences %d ", len(data))

    # save the resulting files as tsv
    train_path = os.path.join(lang_folder, "train.tsv")
    save_tsv_file(train_path, train_sentences)

    test_path = os.path.join(lang_folder, "test.tsv")
    save_tsv_file(test_path, test_sentences)

    # remove unnecessary file: sentences.csv
    os.remove(sent_file)

    # move the clips
    old_clips = os.path.join(input_folder, "clips")
    shutil.move(old_clips, lang_folder)
    logger.info("Clips moved under %s ", lang_folder)

    # save all into tar file
    create_tar_file(input_folder, output_folder)
    logger.info("%s created containing all files ", output_folder)


def preprocess_text(text):
    """
    Preprocess given text
    :param text : str,
    Returns
       str,
    """
    # remove urls
    sequence = prep.remove_urls(text)
    # replace numbers to text
    sequence = prep.replace_numbers(sequence)
    # remove unusual error
    sequence = sequence.replace("--->}}", cnt.EMPTY)
    # replace special characters
    sequence = prep.replace_special_characters(sequence)
    # replace hatted characters
    sequence = prep.replace_hatted_characters(sequence)
    return sequence


def replace_all(text, dic):
    """ replaces all the strings given in the dictionary
    :param text: str, to be replaced
    :param dic: dict, contains strings as keys and their replacements as values
    """
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def generate_text_file(input_file,
                       text_key,
                       output_folder,
                       language="tr",
                       split_mode=None,
                       log_nbr=100):
    """
    Only generates text files for audio data
    Args:
    :param input_file   : str, the path of the input file, json
    :param text_key     : str, the key name of the dict object to read text
    :param language     : str, the abbreviation of the language to be used
    :param output_folder: str, the path of the output folder
    :param split_mode   : str, if not None, it defines how to split the text
           can be word, sentence, None
    :param log_nbr      : int, the number of steps to print
        It checks every audio file if there is an error
    :return
        None
    """
    # reading the input folder
    with open(input_file) as fp:
        data = json.load(fp)
    logger.info("Input file are open from %s" % input_file)

    ind = 0
    text_dict = dict()
    text_out_path = os.path.join(output_folder, "sentences" + cnt.CSV_EXT)

    # output folder for the clips
    output_folder = os.path.join(output_folder, "clips")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for k, v in tqdm(data.items()):
        try:
            # get the text
            text = v[text_key]
            # if text contains non-language characters than no need to process it
            if text:
                # if prep.has_only_language_letters(text) and not(cnt.ERROR_MESSAGE in text):
                # preprocess text
                text = preprocess_text(text)
                if split_mode:
                    text = prep.splitting_text(text, split_mode, language)
                # get the name for the file
                tag_id = replace_all(k, REPLACE_DICT)
                file_name = cnt.NAMETMP.format(tag=prep.normalize_text(v["datafield"]),
                                               name=prep.normalize_text(v["name"]),
                                               tagID=tag_id)
                # where to save audio
                out_file = os.path.join(output_folder, file_name)
                if isinstance(text, list):
                    # if text is split, then there are multiple inputs
                    for i, sequence in enumerate(text):
                        if len(sequence) > 1:
                            sequence = prep.remove_punc(sequence)
                            # out_path = out_file + POINT + str(i) + MP3_EXT
                            # if path exist no need to call tts
                            text_dict[file_name + cnt.POINT + str(i) + cnt.MP3_EXT] = sequence
                elif isinstance(text, str):
                    if text:
                        # removes the punctuations
                        text = prep.remove_punc(text)
                        out_path = out_file + cnt.MP3_EXT
                        # if path exist no need to call tts
                        if not exists(out_path):
                            text_dict[file_name + cnt.MP3_EXT] = text
                ind += 1
                if ind % log_nbr == 0:
                    logger.info("Processed %d numbers of sample" % ind)
                    time.sleep(5)
                    logger.info("Waiting for 10 seconds")
                    write_text_csv(text_dict, text_out_path)
        except SkipTheExample:
            pass

    logger.info("Finished processing %d numbers of sample" % ind)
    # saving the resulting text file
    write_text_csv(text_dict, text_out_path)
    logger.info("Text samples are saved into %s " % text_out_path)

    return


def contain_error_text(text):
    """
    checks if the given text contains error
    :param text:
    :return:
    """
    error_msg_flag = (cnt.ERROR_MESSAGE in text) or ("class=" in text)
    empty_flag = (text == cnt.EMPTY)
    wrong_flag = (re.match(cnt.WRONG_MESSAGE_2, text)) or re.match(cnt.WRONG_MESSAGE, text)
    return wrong_flag or empty_flag or error_msg_flag


def generate_tts_from_file(input_file,
                           text_key,
                           output_folder,
                           language="tr",
                           split_mode=None,
                           log_nbr=100,
                           debugging=False):
    """
    Generates speech data from given text inputs
    Args:
    :param input_file   : str, the path of the input file, json
    :param text_key     : str, the key name of the dict object to read text
    :param language     : str, the abbreviation of the language to be used
    :param output_folder: str, the path of the output folder
    :param split_mode   : str, if not None, it defines how to split the text
           can be word, sentence, None
    :param log_nbr      : int, the number of steps to print
    :param debugging    : bool, if true runs with debugging setting
        It checks every audio file if there is an error
    :return
        None
    """
    # reading the input folder
    with open(input_file) as fp:
        data = json.load(fp)
    logger.info("Input file are open from %s" % input_file)

    ind = 0
    text_dict = dict()
    text_out_path = os.path.join(output_folder, "sentences" + cnt.CSV_EXT)

    # output folder for the clips
    output_folder = os.path.join(output_folder, "clips")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for k, v in tqdm(data.items()):
        try:
            # get the text
            text = v[text_key]
            # if text contains non-language characters than no need to process it
            if text:
                # if prep.has_only_language_letters(text) and not(contain_erreonous_text(text)):
                # preprocess given text
                text = preprocess_text(text)
                if split_mode:
                    text = prep.splitting_text(text, split_mode, language)

                # get the name for the file
                tag_id = replace_all(k, REPLACE_DICT)
                file_name = cnt.NAMETMP.format(tag=prep.normalize_text(v["datafield"]),
                                               name=prep.normalize_text(v["name"]),
                                               tagID=tag_id)
                # where to save audio
                out_file = os.path.join(output_folder, file_name)
                if isinstance(text, list):
                    # if text is split, then there are multiple inputs
                    for i, sequence in enumerate(text):
                        # remove the punctuations before model_tts
                        if len(sequence) > 1:
                            sequence = prep.remove_punc(sequence)
                            out_path = out_file + cnt.POINT + str(i) + cnt.MP3_EXT
                            # if path exist no need to call model_tts
                            if not exists(out_path):
                                try:
                                    tts = gTTS(sequence, lang=language)
                                    tts.save(out_path)
                                    # save text files
                                    text_dict[file_name + cnt.POINT + str(i) + cnt.MP3_EXT] = sequence
                                except TTSError:
                                    pass
                                if debugging:
                                    # test the created file
                                    file_read = test_audio_file(out_path)
                                    if file_read == -1:
                                        raise SkipTheExample()
                            else:
                                logger.info("Path %s exists" % out_path)
                elif isinstance(text, str):
                    if text:
                        out_path = out_file + cnt.MP3_EXT
                        # removes the punctuations before TTS
                        text = prep.remove_punc(text)
                        # if path exist no need to call model_tts
                        if not exists(out_path):
                            try:
                                tts = gTTS(text, lang=language)
                                tts.save(out_path)
                                # save text files
                                text_dict[file_name + cnt.MP3_EXT] = text
                            except TTSError:
                                pass
                        else:
                            logger.info("Path %s exists" % out_path)
                ind += 1
                if ind % log_nbr == 0:
                    logger.info("Processed %d numbers of sample" % ind)
                    time.sleep(5)
                    logger.info("Waiting for 10 seconds")
                    write_text_csv(text_dict, text_out_path)
        except SkipTheExample:
            pass

    logger.info("Finished processing %d numbers of sample" % ind)
    # saving the resulting text file
    write_text_csv(text_dict, text_out_path)
    logger.info("Text samples are saved into %s " % text_out_path)
