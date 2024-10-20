# Audio-Project

This project is Speech-Recognition project using Neural Networks.

---

# Dataset

## Training

The dataset used in this project is [HebDB](https://huggingface.co/datasets/SLPRL-HUJI/HebDB).
As described in the dataset's page, the dataset contains:

A weakly supervised dataset for spoken language processing in the Hebrew language.
HEBDB offers roughly 2500 hours of natural and spontaneous speech recordings in the Hebrew language,
consisting of a large variety of speakers and topics. We provide raw recordings together with a pre-processed,
weakly supervised, and filtered version. The goal of HEBDB is to further enhance research and development of
spoken language processing tools for the Hebrew language.

## Test

We will report **WER** using the [JIWER](https://pypi.org/project/jiwer/) package on
the [FLEURS](https://huggingface.co/datasets/google/fleurs) dataset. More specifically, we will use the
**"test"** split of the on the **"he_il"** subset.

Fleurs is the speech version of the FLoRes machine translation benchmark

---



### Report

For more information about the project, you can check the [report](https://drive.google.com/file/d/1SOEg1jd_2ac5pPWHLe4h7PEMWEHNPUi8/view?usp=sharing) for the project.

Hope you enjoy using the ASR model via the web interface 🔊🔊🔊
