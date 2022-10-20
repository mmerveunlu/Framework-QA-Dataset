import re
import string
import unidecode

from nltk.tokenize import word_tokenize, sent_tokenize
from num2words import num2words
import utils.constants as cnt


def has_only_language_letters(sequence, language="tr"):
    """ Check if the given sequence contains non-language characters
    If so, remove that example.
    :param sequence: string, contains a sentence
    :param language: string, the abbreviation of the language used
    :return bool, True if all the characters are in the alphabet,
                  False if there is at least one character not in the alphabet
    """
    # get the additional characters of the language
    lang_set = cnt.LANGUAGE_CHS[language]
    # concatenate alphabets, numerics and punctuations
    char_set = string.ascii_letters + string.digits + lang_set + string.punctuation + cnt.SPACE
    return all((True if x in char_set else False for x in sequence))


def normalize_text(sequence):
    """
     Normalizes the given sequence:
        removes the space,
        lowers the characters
        replaces non-english characters with english ones
    :param sequence: str, raw text
    :return:
       str, normalized text
    """
    return remove_punc(unidecode.unidecode(sequence.lower())).replace(cnt.SPACE, cnt.EMPTY).replace(cnt.FileSEP, cnt.EMPTY)


def remove_urls(sequence):
    """ removes the urls in the given text
    :param sequence: str, text to be searched for url
    :return:
    """
    return re.sub(r'http\S+', '', sequence)


def replace_numbers(sequence, lang="tr"):
    """ Replaces numbers to text
    :param sequence, string
    :param lang, string, language abbreviations
    :return string.
    """
    # find numbers
    nbr_exp = r'\d+'
    numbers = re.findall(nbr_exp, sequence)
    # replace them with text
    pattern = r'\b{}\b'
    for number in numbers:
        nbr_text = num2words(int(number), lang=lang)
        sequence = re.sub(pattern.format(number), nbr_text, sequence)
    return sequence


def replace_special_characters(sequence):
    """ replace special characters to text
     :param sequence, string
     :return string
     """
    for k, v in cnt.special_character_dict.items():
        sequence = sequence.replace(k, v)
    return sequence


def remove_punc(sequence):
    """ removes the punctuations """
    return re.sub(cnt.chars_to_remove_regex, cnt.SPACE, sequence).lower()


def replace_hatted_characters(sequence):
    """ Replaces hatted characters to non-hatted ones
    :param sequence, string
    :return sequence, string
    """
    sequence = re.sub('[â]', 'a', sequence)
    sequence = re.sub('[î]', 'i', sequence)
    sequence = re.sub('[ô]', 'o', sequence)
    sequence = re.sub('[û]', 'u', sequence)
    return sequence


def split_punctuations(sequence):
    """ split the sequence using punctuaitons: ?,.
    :param sequence, string
    :return string
    """
    return re.split('[?.,]', sequence)


def split_window(sequence, lang="turkish", nbr_word=8):
    """ splits the sequence by window
    :param sequence, string to be split
    :param lang, str, the language that text is from
    :param nbr_word, int, number of max words in window
    :return list of string
    """
    # window based splitting
    words = word_tokenize(sequence, lang)
    split_text = []
    for i in range(0, len(words), nbr_word):
        split_text.append(words[i:i + nbr_word])
    split_text = [" ".join(f) for f in split_text]
    return split_text


def splitting_text(text, split_mode, language="tr"):
    """
    Splits the given text into sentences or words

    :param text       : str, contains raw text to be transformed into speech
    :param split_mode : str, how to split the text, can be: word, sentence, window
    :param language   : str, the abbreviation of the language: tr, eng
    :return: list of string
    """
    if language != "tr":
        raise NotImplementedError("Other languages are not implemented yet!")
    else:
        split_text = text
        if split_mode == "word":
            # word based splitting
            split_text = word_tokenize(text, cnt.LANG_DICT[language])
        elif split_mode == "sentence":
            # sentence based splitting
            split_text = sent_tokenize(text, cnt.LANG_DICT[language])
        elif split_mode == "window":
            split_text = split_window(text, cnt.LANG_DICT[language])
        else:
            raise ValueError("Split mode must be word, sentence or window")
        return split_text

