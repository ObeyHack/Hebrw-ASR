from datasets import load_dataset
from time import time


load_dataset("SLPRL-HUJI/HebDB", "YV_pre", cache_dir='datasets/train', split="train")

load_dataset("google/fleurs", "he_il", split="validation", cache_dir='datasets/val', trust_remote_code=True)

load_dataset("google/fleurs", "he_il", split="test", cache_dir='datasets/test', trust_remote_code=True)