# Running ASR model 
This folder contains functions related to ASR model training and testing. 
The implementation of the model is mostly based on HuggingFace tutorial: 
[Fine-tuning XLS-R for Multi-Lingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)

## Training a model 
This step requires a constructed dataset. 
An example of generated dataset can be found under data/ folder: data/example_tr_wiki/

To train the model, you can use the next command with the example configuration file.

  > python module_asr/run_asr.py --config models/example_model/params.json 

## Predicting from a trained model 
This step requires a constructed dataset and a trained model. 
An example of generated dataset can be found under data/ folder: data/example_tr_wiki/

To test a trained model, we can use a pre-trained one from HuggingFace directly. 
If you have a trained model, you can change the variable "pretrained" in the params-eval.json file.
  
  > python module_asr/run_asr.py --config models/example_model/params-eval.json 
