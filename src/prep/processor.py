from charactertokenizer import CharacterTokenizer
from string import whitespace, punctuation
from hebrew.chars import HEBREW_CHARS
from transformers import WhisperFeatureExtractor
from transformers import Speech2TextFeatureExtractor

FEATURES = 80

def get_tokenizer():
    vocab = [hebrew_char.char for hebrew_char in HEBREW_CHARS] + [" ", ",", ".", "?", "'"] + [f"{i}" for i in range(10)] + [";"]
    model_max_length = 2048
    tokenizer = CharacterTokenizer(vocab, model_max_length)
    return tokenizer


def get_feature_extractor():
    
    #feature_extractor = Speech2TextFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    return feature_extractor
