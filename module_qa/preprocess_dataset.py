"""
The scripts contains preprocessing function to be applied QA datasets before running the model.
The functions: formatting the dataset, splitting the dataset.

Usage:
To format the dataset
 > python preprocess_dataset.py --input train_path.json --output train_final_path.json --preprocess
To split the dataset
 > python preprocess_dataset.py --input train_path.json --output split_data/  --split
 
"""
import argparse
import json
from os.path import join

from tqdm import tqdm
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def generate_qa_examples(data):
    """ returns a list of dict object from given data
    which contains question ids, title, context, question and answer
    :param data : list of dict, has a nested structure to store qa-pairs
       data = {"version":"","data":[{"title","paragraphs":[{"context":"","qas":[{"question":..}]}]]
    returns
       list of dict,
    """
    qa_dataset = []
    for data in tqdm(data["data"]):
        title = data["title"]
        for p in data["paragraphs"]:
            context = p["context"]
            for qa in p["qas"]:
                qid = str(qa["id"])
                q = qa["question"]
                ans = {"text": [], "answer_start": []}
                for answer in qa["answers"]:
                    if answer["text"]:
                        ans["text"].append(answer["text"])
                        ans["answer_start"].append(answer["answer_start"])
                if len(ans["text"]) > 0:
                    example = {"id": qid, "title": title, "context": context, "question": q, "answers": ans}
                    qa_dataset.append(example)
    return qa_dataset


def preprocess_dataset(raw_path, final_path):
    """
    preprocesses the dataset if needed.
    When given raw dataset, it transforms into required format (e.g. SQuAD)
    :param raw_path str, the path of the raw dataset
    :param final_path str, the path of the final dataset
    """
    data_raw = json.load(open(raw_path, "r", encoding="utf-8"))
    data_qa = generate_qa_examples(data_raw)
    # generates new qa dataset in the desired format
    data_full = {"version": "0.1.0", "data": data_qa}
    json.dump(data_full, open(final_path, "w"), indent=4, ensure_ascii=True)


def split_train_test(train_path, out_path, split_rate=0.15, formatted=False):
    """
    This function splits training set and saves
    the resulting file as training and test set
    :param train_path: str, the path of the training path
    :param split_rate: float, the ratio of the split
    :param out_path: str, the output folder path
    :param formatted: boolean, True if the file is formatted using preprocess_dataset
    :return: None
    """
    with open(train_path) as fp:
        data = json.load(fp)
    if formatted:
        data_formatted = data["data"]
    else:
        data_formatted = generate_qa_examples(data)

    train, test = train_test_split(data_formatted, test_size=split_rate, random_state=RANDOM_SEED)
    with open(join(out_path, "train_split.json"), "w+") as fp:
        train_full = {"version": "0.1.0", "data": train}
        json.dump(train_full, fp, indent=4)

    with open(join(out_path, "dev_split.json"), "w+") as fp:
        dev_full = {"version": "0.1.0", "data": test}
        json.dump(dev_full, fp, indent=4)


def assign_ids(data, output_file):
    """
    The function re-assigns ids to the examples
    :param data, dict, contains the examples
    :param output_file, str, the path to save the resulting json file
    """
    new_data = {"data": [], "version": "1.0.0"}
    ind_qa = 0
    for a in data:
        new_a = {"title": a['title'], "paragraphs": []}
        for ps in a["paragraphs"]:
            new_ps = {"context": ps["context"], "qas": []}
            for qa in ps["qas"]:
                new_id = str(qa["id"]) + "_" + str(ind_qa)
                new_qa = {"answers": [], "id": new_id, "question": qa["question"]}
                ind_qa += 1
                for ans in qa["answers"]:
                    new_ans = {"answer_start": ans["answer_start"], "text": ans["text"]}
                    new_qa["answers"].append(new_ans)
                new_ps["qas"].append(new_qa)
            new_a["paragraphs"].append(new_ps)
        new_data["data"].append(new_a)

    with open(output_file, "w+") as fp:
        json.dump(new_data, fp, indent=4)
    return


def parse_args():
    """ Parse the arguments and returns the object """
    parser = argparse.ArgumentParser(description="The script to generate json from text file")
    parser.add_argument("--input",
                        help="Input file as json",
                        required=True)
    parser.add_argument("--output",
                        help="The path of the output file/folder",
                        required=True)
    parser.add_argument("--formatted",
                        help="True if the file was already formatted",
                        action="store_true")
    parser.add_argument("--preprocess",
                        help="True if the file is preprocessed",
                        action="store_true")
    parser.add_argument("--change",
                        help="Set True if ids will be reassigned",
                        action="store_true")
    parser.add_argument("--split",
                        help="Set True if split the dataset",
                        action="store_true")
    parser.add_argument("--ratio",
                        help="The ratio between train/dev split",
                        default=0.15)
    args = parser.parse_args()
    return args


def main():
    """ Runs preprocessing before QA model """
    inputs = parse_args()
    if inputs.preprocess:
        preprocess_dataset(inputs.input, inputs.output)
    if inputs.split:
        split_train_test(inputs.input, inputs.output, inputs.ratio, inputs.formatted)
    if inputs.change:
        with open(inputs.input) as fp:
            data = json.load(fp)
            if 'data' in data:
                data = data['data']
        assign_ids(data, inputs.output)
    return


if __name__ == "__main__":
    main()
