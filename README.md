# Framework-QA-Dataset

This repository contains the official implementation for [A Framework for Automatic Generation of Spoken Question-Answering Data] and proposed dataset TurQuASe.

## About

This repository contains the implementation of a framework to automatically generate a spoken question answering (QA) dataset. 
The framework consists of a question generation (QG) module to generate questions automatically from given text documents, 
a text-to-speech (TTS) module to convert the text documents into spoken form and an automatic speech recognition (ASR) module 
to transcribe the spoken content. The final dataset contains question-answer pairs for both the reference text 
and ASR transcriptions as well as the audio files corresponding to each reference text.

For QG and ASR systems we used pre-trained multilingual encoder-decoder transformer models 
and fine-tuned these models using a limited amount of manually generated QA data 
and TTS-based speech data, respectively. As a proof of concept, 
we investigated the proposed framework for Turkish 
and generated the Turkish Question Answering (TurQuAse) dataset using Wikipedia articles.

## Framework

The framework consists of four parts: (1) Text-to-Speech (TTS) module, Automatic Speech Recognition (ASR) module, 
Question Generation (QG) module, and Question Answering (QA) module. 
The codes for each part can be found under the related module folders.

### Requirements

The models were implemented in Python using [HuggingFace Library](https://huggingface.co/).
For the required libraries please refer to the requirements.txt file. You can download the libraries with the following command.

    pip install -r requirements.txt

### Usage

Here, we explain the usage of the framework by creating a dataset from text collection. 
We assume that you have a textual data that you want to create an SQA dataset from.

1. **Audio data generation with TTS**  
The folder module_tts/ contains functions to generate audio data from given textual data.
The next command runs the necessary scripts at once for an example data: 
> 


3. **Getting transcriptions from ASR**

4. **Generating Questions from QG**

5. **Testing with QA model**

### Models

### Dataset

### Experimental Results

## Citation