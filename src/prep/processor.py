from charactertokenizer import CharacterTokenizer
from string import whitespace, punctuation
from hebrew.chars import HEBREW_CHARS
from transformers import WhisperFeatureExtractor

def get_tokenizer():
    vocab = [hebrew_char.char for hebrew_char in HEBREW_CHARS] + [" "]
    model_max_length = 2048
    tokenizer = CharacterTokenizer(vocab, model_max_length)
    return tokenizer


def get_feature_extractor():
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    return feature_extractor

