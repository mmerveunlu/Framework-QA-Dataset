import json
import random
import os

from IPython.display import display, HTML
import IPython.display as ipd
import pandas as pd

import utils.preprocess_utils as prep


def replace_numbers_batch(batch):
    """ Replaces numbers to text
    :param batch
    """
    batch["sentence"] = prep.replace_numbers(batch["sentence"])
    return batch


def show_random_elements(dataset, num_examples=10):
    """
    Shows random examples from dataset
    :param dataset, Dataset object
    :param num_examples, int, number of examples to be listed
    """
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))


def remove_special_characters(batch):
    """
    Deletes defined characters from sentence
    :param batch, an example with ['path','sentence','audio']
    :return batch
    """
    batch["sentence"] = prep.remove_punc(batch['sentence'])
    return batch


def replace_hatted_characters(batch):
    """
    replaces hatted characters with non-hatted ones
    :param batch, example object
    :return batch, example object
    """
    batch['sentence'] = prep.replace_hatted_characters(batch['sentence'])
    return batch


def extract_all_chars(batch):
    """
    Collects unique characters from dataset
    :param batch: an example from Dataset
    :return: dict, contains vocab and sentences
    """
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def play_audio(dataset):
    """ Randomly selects and plays one audio
    :param dataset: Dataset object
    """
    rand_int = random.randint(0, len(dataset) - 1)

    print(dataset[rand_int]["sentence"])
    ipd.Audio(data=dataset[rand_int]["audio"]["array"], autoplay=True, rate=16000)


def get_vocab(train_set, test_set, output_folder):
    """
    Saves the character vocabulary from train and test set
    :param train_set, dataset object
    :param test_set, dataset object
    :param output_folder, str, path to save vocabulary file
    """
    # collects vocabulary
    vocab_train = train_set.map(extract_all_chars,
                                batched=True,
                                batch_size=-1,
                                keep_in_memory=True,
                                remove_columns=train_set.column_names)

    vocab_test = test_set.map(extract_all_chars,
                              batched=True,
                              batch_size=-1,
                              keep_in_memory=True,
                              remove_columns=test_set.column_names)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    # change space
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    # adding UNK and PAD tokens
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    # saving the vocabulary
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    vfile = os.path.join(output_folder, "vocab.json")
    with open(vfile, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    return vocab_dict

