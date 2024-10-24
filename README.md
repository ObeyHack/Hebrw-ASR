# Audio-Project

This project is Speech-Recognition project using Neural Networks. 
The architecture used is similar to the DeepSpeech2 model as described in the paper [Deep Speech 2](https://arxiv.org/abs/1512.02595).

---
# Usage

1. install the required packages using the following command:
```bash
pip install .
```
2. Run the following command to start the web interface:
```bash
app
```
you may to install the model.bin.lm file using the following commands in 
data_prep.py 

---
# Dataset

## Training

The dataset used in this project is [whisper-training](https://huggingface.co/datasets/ivrit-ai/whisper-training).
The dataset is a collection of Hebrew speech data, and their corresponding transcriptions.

## Test

We will report **WER** using the [JIWER](https://pypi.org/project/jiwer/) package on
the [FLEURS](https://huggingface.co/datasets/google/fleurs) dataset. More specifically, we will use the
**"test"** split of the on the **"he_il"** subset.

---
### Report
For more information about the project, you can check the [report](https://drive.google.com/file/d/1SOEg1jd_2ac5pPWHLe4h7PEMWEHNPUi8/view?usp=sharing) for the project.

Hope you enjoy using the ASR model via the web interface 🔊🔊🔊
