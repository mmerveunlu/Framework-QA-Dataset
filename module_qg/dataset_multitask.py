""" Turkish-Dataset Multitask: This class is based on HuggingFace SQuAD class."""

from __future__ import absolute_import, division, print_function

import json
import logging
import os

import nltk

nltk.download('punkt')

import nlp

_CITATION = """\
"""

_DESCRIPTION = """\
"""

QG_FORMATS = [
    "prepend",
    "highlight",
    "prepend_highlight",
]


class TrDatasetMultitaskConfig(nlp.BuilderConfig):
    """BuilderConfig for TrDataset."""

    def __init__(self, data_dir, data_files, qg_format="highlight", **kwargs):
        """BuilderConfig for TrDataset.

    Args:
      **kwargs: keyword arguments forwarded to super.
    """
        super(TrDatasetMultitaskConfig, self).__init__(**kwargs)
        self.qg_format = qg_format
        self.data_dir = data_dir
        self.data_files = data_files


class TrMultitask(nlp.GeneratorBasedBuilder):
    """ """

    data_dir = "data/"
    data_files = {'train': "train.json", 'validation': "dev.json"}

    BUILDER_CONFIGS = [
        TrDatasetMultitaskConfig(
            name=f"{format_}_qg_format",
            version=nlp.Version("1.0.0", "New split API (https://tensorflow.org/datasets/splits)"),
            description="Plain text",
            data_dir="data/",
            data_files={'train': "train.json", 'validation': "dev.json"},
            qg_format=format_,
        )
        for format_ in QG_FORMATS
    ]

    print(BUILDER_CONFIGS)

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    "source_text": nlp.Value("string"),
                    "target_text": nlp.Value("string"),
                    "task": nlp.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="",
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            "train": os.path.join(self.data_dir, self.data_files["train"]),
            "dev": os.path.join(self.data_dir, self.data_files["validation"]),
        }
        downloaded_files = dl_manager.extract(urls_to_download)
        print(downloaded_files)
        print(nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}))
        return [
            nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _get_correct_alignement(self, context, answer):
        """ Some original examples in dataset have indices wrong by 1 or 2 character. We test and fix this here. """
        gold_text = answer['text']
        start_idx = int(answer['answer_start'])
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx  # When the gold label position is good
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            return start_idx - 1, end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            return start_idx - 2, end_idx - 2  # When the gold label is off by two character
        else:
            return -1, -1

    def process_qa_text(self, context, question, answer):
        ans_gen_input = f"question: {question}  context: {context}"
        ans_gen_target = f"{answer}"
        return {"source_text": ans_gen_input, "target_text": ans_gen_target, "task": "qa"}

    def process_qg_text(self, context, question, answer):
        answer_text = answer['text'].strip()

        if self.config.qg_format == "prepend":
            que_gen_input = f"answer: {answer_text}  context: {context}"
        elif self.config.qg_format == "highlight":
            start_pos, end_pos = self._get_correct_alignement(context, answer)
            que_gen_input = f"generate question: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"
        else:
            start_pos, end_pos = self._get_correct_alignement(context, answer)
            que_gen_input = f"answer: {answer_text} context: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"

        que_gen_target = f"{question}"
        return {"source_text": que_gen_input, "target_text": que_gen_target, "task": "qg"}

    def process_e2e_qg(self, paragraph):
        source_text = f"generate questions: {paragraph['context'].strip()}"
        questions = [qas['question'].strip() for qas in paragraph['qas']]
        target_text = " {sep_token} ".join(questions)
        target_text = f"{target_text} {{sep_token}}"
        return {"source_text": source_text, "target_text": target_text, "task": "e2e_qg"}

    def process_ans_ext(self, paragraph):
        context = paragraph['context'].strip()

        # split into sentences
        sents = nltk.sent_tokenize(context)

        # get positions of the sentences
        positions = []
        for i, sent in enumerate(sents):
            if i == 0:
                start, end = 0, len(sent)
            else:
                start, end = (prev_end + 1), (prev_end + len(sent) + 1)
            prev_end = end
            positions.append({'start': start, 'end': end})

        # get answers
        answers = [qa['answers'][0] for qa in paragraph['qas']]

        # get list of answers for each sentence
        sent_answers = []
        for pos, sent in zip(positions, sents):
            target_answers = []
            for ans in answers:
                if ans['answer_start'] in range(pos['start'], pos['end']):
                    target_answers.append(ans['text'].strip())
            sent_answers.append(target_answers)

        # build inputs and targets
        examples = []
        for i, ans in enumerate(sent_answers):
            context = "extract answers:"
            if len(ans) == 0: continue
            ans = list(set(ans))
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "{hl_token} %s {hl_token}" % sent
                context = "%s %s" % (context, sent)
                context = context.strip()
            input_text = context
            target_text = " {sep_token} ".join(ans) + " {sep_token}"

            examples.append({'source_text': input_text, "target_text": target_text, "task": "ans_ext"})

        return examples

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        count = 0
        tasks = ['qa', 'qg', 'ans_ext', 'e2e_qg']
        with open(filepath) as f:
            squad = json.load(f)
            for article in squad["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()

                    if 'ans_ext' in tasks:
                        ans_ext_examples = self.process_ans_ext(paragraph)
                        for example in ans_ext_examples:
                            yield count, example
                            count += 1

                    if 'e2e_qg' in tasks:
                        yield count, self.process_e2e_qg(paragraph)
                        count += 1

                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answers = [answer["text"].strip() for answer in qa["answers"]]
                        for task in tasks:
                            if task == 'qa':
                                yield count, self.process_qa_text(context, question, answers[0])
                                count += 1

                            if task == 'qg':
                                yield count, self.process_qg_text(context, question, qa["answers"][0])
                                count += 1
