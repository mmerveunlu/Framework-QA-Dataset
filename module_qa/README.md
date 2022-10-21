# Training a QA model 
This module contains function to run a question answering model.
The scripts are based on official examples of [HuggingFace Question Answering](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering).


1. If needed, we can split the dataset into train/dev or preprocess the dataset.
It is needed when the given dataset format is different from expected.
Next command takes a dataset which is in nested format, and returns a list of dict. 

> python module_qa/preprocess_datasets.py \
   --input data/example_sqa.json  \
   --output data/example_sqa_formatted.json \
   --preprocess 

2. To train a model on the dataset, we can run the next command. 
> python module_qa/run_qa.py \
  --model_name_or_path bert-base-uncased \
  --output_dir models/example_model_qa  \
  --train_file data/example_sqa_formatted.json \
  --validation_file data/example_sqa_formatted.json \
  --test_file data/example_sqa_formatted.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --load_best_model_at_end \
  --metric_for_best_model eval_f1 \ 
  --save_strategy epoch \
  --save_total_limit 2 \
  --evaluation_strategy epoch 

