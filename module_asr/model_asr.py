import json
from os.path import join

from datasets import load_dataset, load_metric
from mutagen.mp3 import MP3
import numpy as np
import torch
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer

import prepare_dataset_asr as pda
import data_collator_ctc as dcc


def get_length(batch):
    """ returns the length of the audio"""
    audio = MP3(batch['path'])
    length = audio.info.length
    batch['length'] = length
    return batch


def load_dataset_from(data_script, part_name, split_name, replace_nbrs=False):
    """
    Loads the dataset from given dataset script
    :param data_script, str path of the script
    :param part_name, str the name of the used part as part1, part2
    :param split_name, str, can be train, test or valid
    :param replace_nbrs, bool, set True if numbers are replaced by text
    """
    dataset = load_dataset(data_script, part_name, split=split_name)
    # removes special characters
    dataset = dataset.map(pda.remove_special_characters)
    if replace_nbrs:
        dataset = dataset.map(pda.replace_numbers_batch)
    # removes hatted characters from sentences
    dataset = dataset.map(pda.replace_hatted_characters)
    return dataset


def train_model(params):
    """
    trains the given model
    :param params, MyParams object
     """
    # "facebook/wav2vec2-xls-r-300m",
    # "spoken_wiki.py"
    dataset_script = params.dataset_script

    # loads the dataset
    if params.cross_validation:
        train_set = load_dataset_from(dataset_script, params.part_name, split_name="train")
        test_set = load_dataset(dataset_script, params.part_name, split="test[50%:]")
    elif params.split_train_data:
        train_set = load_dataset_from(dataset_script, params.part_name, split_name="train[-90%:]")
        test_set = load_dataset_from(dataset_script, params.part_name, split_name="train[:10%]")
    else:
        train_set = load_dataset_from(dataset_script, params.part_name, split_name="train")
        test_set = load_dataset_from(dataset_script, params.part_name, split_name="test")
    # get character vocabulary
    pda.get_vocab(train_set, test_set, params.output_dir)

    #  defines the tokenizer
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(params.output_dir,
                                                     unk_token=params.unk_token,
                                                     pad_token=params.pad_token,
                                                     word_delimiter_token=params.word_delimiter)
    # defines the feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=params.feature_size,
                                                 sampling_rate=params.sampling_rate,
                                                 padding_value=params.padding_value,
                                                 do_normalize=params.do_normalize,
                                                 return_attention_mask=True)
    # defines the processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                  tokenizer=tokenizer)

    data_collator = dcc.DataCollatorCTCWithPadding(processor=processor, padding=True)

    def prepare_dataset(batch):
        """ Prepares dataset
        :param batch, one example from Dataset
        :return batch, tokenized example
        """
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"],
                                          sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    train_set = train_set.map(prepare_dataset, remove_columns=train_set.column_names)
    test_set = test_set.map(prepare_dataset, remove_columns=test_set.column_names)
    # model, training_args = prepare_model_Wav2Vec(pretrained_model_path, processor)

    model = Wav2Vec2ForCTC.from_pretrained(
        params.pretrained_model_path,
        attention_dropout=params.attention_dropout,
        hidden_dropout=params.hidden_dropout,
        feat_proj_dropout=params.feat_proj_dropout,
        mask_time_prob=params.mask_time_prob,
        layerdrop=params.layerdrop,
        ctc_loss_reduction=params.ctc_loss_reduction,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        group_by_length=params.group_by_length,
        output_dir=params.output_dir,
        per_device_train_batch_size=params.per_device_train_batch_size,
        per_device_eval_batch_size=params.per_device_train_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        evaluation_strategy=params.evaluation_strategy,
        num_train_epochs=params.num_train_epochs,
        gradient_checkpointing=params.gradient_checkpointing,
        fp16=params.fp16,
        save_steps=params.save_steps,
        eval_steps=params.eval_steps,
        logging_steps=params.logging_steps,
        learning_rate=params.learning_rate,
        warmup_steps=params.warmup_steps,
        save_total_limit=params.save_total_limit,
        push_to_hub=params.push_to_hub,
    )

    def compute_metrics(pred):
        """
        Computes the measure using the prediction
        :params pred, model output
        :params processor, Processor object
        :return dict, contains metric
        """
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer_metric = load_metric("wer")
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # run trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_set,
        eval_dataset=test_set,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    print("Training is done")
    trainer.save_model()
    # Need to save processor
    processor.save_pretrained(params.output_dir)
    trainer.evaluate()


def evaluate_model(params):
    """ make predictions using saved model """
    # load trained model
    model = Wav2Vec2ForCTC.from_pretrained(params.output_dir).to("cuda")
    processor = Wav2Vec2Processor.from_pretrained(params.output_dir)

    # load dataset
    # eval_set = load_dataset(params.dataset_script, params.part_name, split="test[-50%:]")
    eval_set = load_dataset_from(params.dataset_script, params.part_name, params.eval_split_name)

    # prepare dataset
    def prepare_dataset(batch):
        """ Prepares dataset
        :param batch, one example from Dataset
        :return batch, tokenized example
        """
        audio = batch["audio"]

        # batched output is "un-batched"
        batch["input_values"] = processor(audio["array"],
                                          sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    # prepare dataset as training
    eval_set = eval_set.map(prepare_dataset, remove_columns=['audio'])

    def evaluate(batch):
        # prepare the inputs
        inputs = processor(batch["input_values"],
                           sampling_rate=params.sampling_rate,
                           return_tensors="pt",
                           padding=True)
        # predicts
        with torch.no_grad():
            logits = model(inputs.input_values.to("cuda"),
                           attention_mask=inputs.attention_mask.to("cuda")).logits

        pred_ids = torch.argmax(logits, dim=-1)
        # decode the predictions to string
        batch["pred_strings"] = processor.batch_decode(pred_ids)
        return batch

    # then evaluate the model
    result_eval = eval_set.map(evaluate, batched=True, batch_size=params.per_device_train_batch_size)

    # then calculate WER
    # Loading metric
    wer = load_metric("wer")

    eval_wer = 100 * wer.compute(predictions=result_eval["pred_strings"], references=result_eval['sentence'])

    # save the results and the predictions
    # cols_to_remove = ['path', 'sentence', 'pred_strings']
    cols_to_remove = ['labels', 'input_values']
    result_eval = result_eval.remove_columns(cols_to_remove)

    result_eval.save_to_disk(join(params.output_dir, "prediction_on_test_" + params.part_name))
    # save wer rates
    result = {"eval_wer": eval_wer}
    with open(join(params.output_dir, "wer_results_" + params.part_name + ".json"), "w+") as fp:
        json.dump(result, fp)
