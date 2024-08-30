import os
from litdata import optimize
from datasets import Dataset, Audio, load_dataset, load_from_disk
from functools import partial
from datasets import disable_caching
from processor import get_tokenizer, get_feature_extractor, FEATURES


def is_text(text):
    # Filter out examples without text (text is empty or None)
    return text is not None and len(text) > 0


def is_audio(audio):
    # Filter out examples without audio (audio is empty or None)
    return audio["array"] is not None and len(audio["array"]) > 0


def tokenization(examples, tokenizer):
    return tokenizer(examples["normalized_text"], padding="max_length")


def feature_extraction(example, feature_extractor):
    audio = example["audio"]
    example["mfcc"] = (feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], 
                                        ).input_features[0]).T
    return example


def pre_process(dataset):
    """
    Pre-process the data
    """

    dataset = dataset.take(2)

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

    dataset = dataset.with_format("torch")
    for item in dataset:
        yield {"mfcc": item["mfcc"], "input_ids": item["input_ids"]}



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

    #dataset_train = load_dataset("SLPRL-HUJI/HebDB", "YV_pre", cache_dir='datasets/train', split="train")
    # output_dir_train = f"{output_root}/train/YV"
    # optimizer(dataset_train, output_dir_train)


    dataset_train = load_dataset("ivrit-ai/audio-labeled",  cache_dir='datasets/train', split="train")
    #output_dir_train = f"{output_root}/train/ivrit-ai"
    output_dir_train = f"preprocess/train/ivrit-ai"
    dataset_train = dataset_train.rename_column("orig_text", "normalized_text")
    optimizer(dataset_train, output_dir_train)

    # dataset_val = load_dataset("google/fleurs", "he_il", split="validation", cache_dir='datasets/val', trust_remote_code=True)
    # dataset_val = dataset_val.rename_column("transcription", "normalized_text")
    # output_dir_val = f"{output_root}/val"
    # optimizer(dataset_val, output_dir_val)
    
    # dataset_test = load_dataset("google/fleurs", "he_il", split="test", cache_dir='datasets/test', trust_remote_code=True)
    # dataset_test = dataset_test.rename_column("transcription", "normalized_text")
    # output_dir_train = f"{output_root}/test"
    # optimizer(dataset_test, output_dir_train)


main()