"""
This script applies filtering to given json file.
The json file contains text data. The filtering steps are:
 * The examples whose paragraphs contain more words than max_word and fewer than min_word are removed.
 * The examples whose type column is erroneous or empty will be removed.
 * The examples from most frequent types are chosen
Args:
    input    : str, contains json dump
    output   : str, where to save resulting dict, should be json also
    max_word : int, maximum number of words in a paragraph
    min_word : int, the minimum number of words in a paragraph
    non_type : bool, Set True if examples with emtpy or erroneous type column will be removed
    most_freq: bool, Set True if only most frequent types are chosen
    number   : int, the number of examples chosen from each class
"""

import argparse
import json
import logging
import pandas as pd

logging.basicConfig(filename='log/data.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)


def parse_args():
    """ Parse the arguments and returns the object """
    parser = argparse.ArgumentParser(description="The script to generate json from text file")
    parser.add_argument("--input",
                        help="Input file as text",
                        required=True)
    parser.add_argument("--output",
                        help="The path of the output folder",
                        required=True)
    parser.add_argument("--word_flag",
                        help="Set True, if word-length filtering is applied",
                        action="store_true")
    parser.add_argument("--max_word",
                        help="The maximum number of words",
                        default=100)
    parser.add_argument("--min_word",
                        help="The minimum number of words",
                        default=4)
    parser.add_argument("--non_type",
                        help="Set if non-type examples will be removed",
                        action="store_true")
    parser.add_argument("--most_freq",
                        help="Set if only most frequent types will be chosen",
                        action="store_true")
    parser.add_argument("--number",
                        help="Number of example chosen from each type",
                        default=500)
    args = parser.parse_args()
    return args


def choose_most_frequent_types(df_all, most_freq_cnt=500, less_freq_cnt=10):
    """
    Finds the most frequent types in the data and returns them
    :param less_freq_cnt: int, the maximum frequency number for a type to be considered as less frequent
    :param most_freq_cnt: int, the minimum frequency number for a type to be considered as most frequent
    :param df_all: dataframe, contains data to get frequency
    :return:
     df, contains less frequent types
     df, contains most frequent types
    """
    # Get the type counts
    df_types = df_all.groupby(['datafield']).agg('count').reset_index()
    # Remove unneccesary columns
    df_types = df_types.drop(['name', 'table', 'paragraph', 'cnt_word'], axis=1)
    # Rename the colum as counts
    df_types = df_types.rename(columns={"number": "counts"})
    df_types = df_types.sort_values(by=['counts'])

    # Get the most frequent ones
    less_frequent = df_types[df_types['counts'] < less_freq_cnt]
    most_frequent = df_types[df_types['counts'] >= most_freq_cnt]

    return less_frequent, most_frequent


def save_from_each_class(data_frame, output_path, number=500):
    """
    Chooses examples from each class and saves dataframe as dict
    :param data_frame, pandas Dataframe contains data
    :param output_path, str, where to save the result
    :param number, int, the number of examples to collect from each class
    :return None
    """
    data_frame = data_frame.groupby('datafield').apply(
        lambda x: x[:number][['number', 'name', 'datafield', 'table', 'paragraph']])

    # transform dataframe into a dict
    sub_dict = data_frame.to_dict()
    data_dict = {}
    for k, v in sub_dict.items():
        for k1, v1 in v.items():
            orig_key = k1[1]
            if not (orig_key in data_dict):
                data_dict[orig_key] = {}
            data_dict[orig_key][k] = v1

    logger.info("There are %d numbers of example in the dictionary " % len(data_dict))
    # save the new dict
    with open(output_path, "w") as fp:
        json.dump(data_dict, fp, indent=4)
    logger.info("The resulting data is saved to %s" % output_path)


def run_filter_data(input_path,
                    max_word_flag=True,
                    max_word_nbr=100,
                    min_word_nbr=4,
                    remove_non_type=True,
                    most_freq_flag=True):
    """
    Filters the given data w.t.r. to the configurations
    :param max_word_flag  : bool, set True if a filtering applied by the max-min word numbers
    :param most_freq_flag : bool, set True if a filtering applied by choosing the most frequent types
    :param remove_non_type: bool, set True if examples w/o types will be removed
    :param min_word_nbr   : int, the minimum number of words in the paragraph after filtering
    :param max_word_nbr   : int, the maximum number of words in the paragraph after filtering
    :param input_path     : str, the path of the input json file
    :return:
     dataframe, contains filtered data
    """

    with open(input_path) as fp:
        data = json.load(fp)
    logger.info("There are %d numbers of example " % len(data))
    # Transform into pandas dataframe
    df_all = pd.DataFrame.from_dict(data, orient="index")

    # Filter by word number
    # Get the length of the passages by characters
    # cnt_character = df_all['paragraph'].map(lambda calc: len(calc))
    # df_all['cnt_character'] = cnt_character

    if max_word_flag:
        # Get the word number
        cnt_word = df_all["paragraph"].apply(lambda x: len(str(x).split(' ')))
        df_all['cnt_word'] = cnt_word
        df_all = df_all[(df_all['cnt_word'] <= max_word_nbr) & (df_all['cnt_word'] > min_word_nbr)]
        logger.info("After filtering by word number, Numbers of example: %d " % len(df_all))

    # What is the distribution of types ?
    types = []
    error_types = []
    empty_types = []
    for i in range(len(df_all)):
        field = df_all.iloc[i].datafield
        if isinstance(field, dict):
            empty_types.append(df_all.iloc[i].number)
        else:
            if "{{" in field:
                error_types.append(df_all.iloc[i].number)
            else:
                types.append(df_all.iloc[i].number)

    # How about the numbers
    logger.info("Total example %d" % len(df_all))
    logger.info("Examples with empty type field %d" % len(empty_types))
    logger.info("Examples with erroneous fields %d" % len(error_types))
    logger.info("Examples with correct fields %d" % len(types))
    # logger.info("Total sum: " % (len(empty_types) + len(error_types) + len(types)))

    if remove_non_type:
        df_all = df_all[df_all['number'].isin(types)]
        logger.info("After filtering by correct type assignment, Numbers of example: %d " % len(df_all))

    if most_freq_flag:
        less_freqs, most_freqs = choose_most_frequent_types(df_all)
        # select examples with most_frequent types
        most_freq_type = list(most_freqs['datafield'])
        df_all = df_all[df_all['datafield'].isin(most_freq_type)]
        logger.info("After choosing the most frequent types: the number of ex %d " % len(df_all))

    return df_all


def main():
    """
        Runs all the process at once.
        :return:
            None
    """
    # Get the inputs
    inputs = parse_args()
    data_frame = run_filter_data(inputs.input,
                                 max_word_flag=inputs.word_flag,
                                 max_word_nbr=int(inputs.max_word),
                                 min_word_nbr=int(inputs.min_word),
                                 remove_non_type=inputs.non_type,
                                 most_freq_flag=inputs.most_freq)
    logger.info("Filtering is done.")
    save_from_each_class(data_frame,
                         inputs.output,
                         number=inputs.number)
    logger.info("Choosing is done.")
    return


if __name__ == "__main__":
    main()
