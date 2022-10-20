"""
This script runs ASR model with given configurations.
It requires a json configuration file to get the parameters.
An example of json file is under models/example_model/params.json

Usage:
> python run_asr.py --config models/example_model/params.json

Args:
  config: str, the path of the configuration file
"""

import argparse
import json
import logging

from model_asr import train_model, evaluate_model


logging.basicConfig(filename='log/data.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)


class MyParams:
    """ Parsing the given configuration """
    def __init__(self, configs):
        self.mode = configs["mode"]
        self.part_name = configs["dataset"]["part_name"]
        self.output_dir = configs["model"]["output_dir"]
        self.model_name = configs["model"]["model_name"]
        self.pretrained_model_path = configs["model"]['pretrained']
        self.dataset_script = configs["model"]['dataset_script']
        self.attention_dropout = configs["model"]["attention_dr"]
        self.hidden_dropout = configs["model"]["hidden_dr"]
        self.feat_proj_dropout = configs["model"]["feat_proj_dr"]
        self.mask_time_prob = configs["model"]["mask_time_prob"]
        self.layerdrop = configs["model"]["layer_drop"]
        self.ctc_loss_reduction = configs["model"]["ctc_loss_reduction"]
        self.cross_validation = configs["model"]["cross_validation"]
        self.split_train_data = configs["model"]["split_train_data"]
        self.eval_split_name = configs["model"]["eval_split_name"]
        self.unk_token = configs["tokenizer"]['unk']
        self.pad_token = configs["tokenizer"]['pad']
        self.word_delimiter = configs["tokenizer"]['word_delimited']
        self.sampling_rate = configs["extractor"]['sampling_rate']
        self.padding_value = configs["extractor"]['padding_value']
        self.feature_size = configs["extractor"]["feature_size"]
        self.do_normalize = configs["extractor"]["normalize"]

        self.group_by_length = configs["training_args"]["group_by_length"]
        self.per_device_train_batch_size = configs["training_args"]["batch_size"]
        self.gradient_accumulation_steps = configs["training_args"]["gradient_acc_steps"]
        self.evaluation_strategy = configs["training_args"]["evaluation_strategy"]
        self.num_train_epochs = configs["training_args"]["epochs"]
        self.gradient_checkpointing = configs["training_args"]["gradient_chck"]
        self.fp16 = configs["training_args"]["fp16"]
        self.save_steps = configs["training_args"]["save_steps"]
        self.eval_steps = configs["training_args"]["eval_steps"]
        self.logging_steps = configs["training_args"]["log_steps"]
        self.learning_rate = configs["training_args"]["lr"]
        self.warmup_steps = configs["training_args"]["warmup_steps"]
        self.save_total_limit = configs["training_args"]["save_total_limit"]
        self.push_to_hub = configs["training_args"]["push_to_hub"]


def parse_args():
    """ Argument parser"""
    parser = argparse.ArgumentParser(description='Train/Test pre-trained model')
    parser.add_argument('--config',
                        help='config file name',
                        required=True)
    args = parser.parse_args()
    return args


def main():
    """ Runs training/testing the model using configuration file """
    # Get the inputs
    args = parse_args()
    with open(args.config) as fp:
        configs = json.load(fp)
    logger.info("Running the model with configuration %s", args.config)
    # transform configs into params
    params = MyParams(configs)
    if params.mode == "train":
        # training a new model
        logger.info("Training started!")
        train_model(params)
    if params.mode == "eval":
        # evaluate the model
        logger.info("Evaluation started!")
        evaluate_model(params)
    return


if __name__ == "__main__":
    main()
