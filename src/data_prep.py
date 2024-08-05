import os
from litdata import optimize
from transformers import AutoTokenizer
from datasets import Dataset, Audio
from transformers import WhisperFeatureExtractor


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
    example["mfcc"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    return example


def pre_process(file_path):
    """
    Pre-process the data
    :param file_path: .arrow file path
    :return:
    """
    dataset = Dataset.from_file(file_path)

    # testing
    dataset = dataset.take(10)

    # Filter out examples without text or audio
    dataset = dataset.filter(is_text, input_columns="normalized_text").filter(is_audio, input_columns="audio")

    # Cast the audio column to Audio type
    sr = 16000
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))

    # Tokenize the text
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    dataset = dataset.map(tokenization, fn_kwargs={"tokenizer": tokenizer}, batched=True)

    # Feature extraction - mfcc
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    dataset = dataset.map(feature_extraction,  fn_kwargs={"feature_extractor": feature_extractor}, batched=False)

    # Drop columns
    dataset = dataset.remove_columns(["audio", "fname", "text", "score", "raw", "is_raw", "source", "n_samples",
                                      "normalized_text", "attention_mask"])

    dataset = dataset.with_format("torch")
    yield {"mfcc": dataset["mfcc"], "input_ids": dataset["input_ids"]}



def main():
    input_dir = r"""data"""
    arrow_files = [os.path.join(root, f) for root, _, filenames in os.walk(input_dir) for f in filenames if
                   f.endswith(".arrow") and f.startswith("data")]

    # for i, file in enumerate(arrow_files):
    #     dataset = Dataset.from_file(file)
    #     print(i)
    #     print(dataset.features)
    #     print("#" * 100)

    # pre_process(arrow_files[3])   # Test the pre_process function

    outputs = optimize(
        fn=pre_process,
        inputs=arrow_files,
        output_dir="./test/",
        chunk_bytes="64MB",
    )


if __name__ == '__main__':
    main()