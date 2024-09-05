import os
from litdata import optimize
from datasets import Dataset, Audio, load_dataset, load_from_disk
from functools import partial
from datasets import disable_caching
from processor import get_tokenizer, get_feature_extractor, FEATURES, MAX_TOKENS, MAX_TIME_STEPS
import speechpy
import torch

def is_text(text):
    # Filter out examples without text (text is empty or None)
    return text is not None and len(text) > 0


def is_audio(audio):
    # Filter out examples without audio (audio is empty or None)
    return audio["array"] is not None and len(audio["array"]) > 0


def tokenization(examples, tokenizer):
    tokenized =  tokenizer(examples["normalized_text"], 
                        padding="max_length", 
                        truncation=False,
                        return_tensors="pt")

    examples["text_encoded"] = tokenized["input_ids"]
    examples["text_length"] = tokenized["attention_mask"].sum(dim=1, keepdim=True).int()
    return examples


def feature_extraction(example, feature_extractor):
    audio = example["audio"]
    fs = audio["sampling_rate"]
    mfcc = feature_extractor(audio["array"], 
                        sampling_rate=audio["sampling_rate"], 
                        do_normalize=True,
                        padding="max_length", 
                        truncation=False,
                        return_tensors="pt",                        
                        # return_attention_mask = True,
                        return_token_timestamps = True,
                        )

    example["mfcc"] = mfcc["input_features"].squeeze(0)
    example["mfcc_length"] = mfcc["num_frames"].int()
    return example


def pre_process(dataset):
    """
    Pre-process the data
    """

    # Filter out examples without text or audio
    dataset = dataset.filter(is_text, input_columns="normalized_text").filter(is_audio, input_columns="audio")

    # Cast the audio column to Audio type
    sr = 16000
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))

    # Tokenize the text
    tokenizer = get_tokenizer()
    dataset = dataset.map(tokenization, fn_kwargs={"tokenizer": tokenizer}, batched=True)

    # Feature extraction - mfcc
    feature_extractor = get_feature_extractor()
    dataset = dataset.map(feature_extraction,  fn_kwargs={"feature_extractor": feature_extractor}, batched=False)
    
    # convert to torch tensors
    dataset.set_format("torch")

    # filter out examples with too long text or audio
    dataset = dataset.filter(lambda x: x["text_length"] < MAX_TOKENS and len(tokenizer.decode(x["text_encoded"], 
                                                                                skip_special_tokens=True).strip(" ")) > 0)
    dataset = dataset.filter(lambda x: x["mfcc_length"] < MAX_TIME_STEPS and x["mfcc_length"] > 0)

    for item in dataset:
        yield {"mfcc": item["mfcc"],
            "mfcc_length": item["mfcc_length"],
            "text_encoded": item["text_encoded"],
            "text_length": item["text_length"],
        }



def optimizer(dataset, output_dir):
    num_of_processes = os.cpu_count()
    d = 4
    datasets = [dataset.shard(d*num_of_processes, i) for i in range(d*num_of_processes)]
    optimize(
        fn=pre_process,
        inputs=datasets,
        output_dir=output_dir,
        chunk_bytes="64MB",
        num_workers=num_of_processes,)



def main():
    output_root = "/teamspace/s3_connections/audio-speech-hebrew"

    # dataset_train = load_dataset("SLPRL-HUJI/HebDB", "YV_pre", cache_dir='datasets/train', split="train")
    # output_dir_train = f"{output_root}/train/YV_norm"
    # optimizer(dataset_train, output_dir_train)


    # dataset_train = load_dataset("ivrit-ai/audio-labeled",  cache_dir='datasets/train', split="train")
    # output_dir_train = f"{output_root}/train/ivrit-ai"
    # # output_dir_train = f"preprocess/train/data"
    # dataset_train = dataset_train.rename_column("text", "normalized_text")
    # dataset_train = dataset_train.take(10000)
    # optimizer(dataset_train, output_dir_train)


    # dataset_train = load_dataset("imvladikon/hebrew_speech_kan",  cache_dir='datasets/train', split="train")
    # output_dir_train = f"{output_root}/train/hebrew_speech_kan-ai"
    # # output_dir_train = f"preprocess/train/data3"
    # # dataset_train = dataset_train.take(100)
    # dataset_train = dataset_train.rename_column("sentence", "normalized_text")
    # optimizer(dataset_train, output_dir_train)

    dataset_val = load_dataset("google/fleurs", "he_il", split="validation", cache_dir='datasets/val', trust_remote_code=True)
    dataset_val = dataset_val.rename_column("transcription", "normalized_text")
    output_dir_val = f"{output_root}/val/"
    optimizer(dataset_val, output_dir_val)
    
    dataset_test = load_dataset("google/fleurs", "he_il", split="test", cache_dir='datasets/test', trust_remote_code=True)
    dataset_test = dataset_test.rename_column("transcription", "normalized_text")
    output_dir_train = f"{output_root}/test/"
    optimizer(dataset_test, output_dir_train)


main()