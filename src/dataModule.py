import lightning as pl
from litdata import StreamingDataLoader, StreamingDataset
from prep.processor import get_feature_extractor, get_tokenizer
import os

CLASSES = get_tokenizer().vocab_size + 1 - 7 # 1 for epsilon, 7 for special tokens (pad, cls, sep, etc)
FEATURES = 80
BATCH_SIZE = 32

class SpeechStreamingDataset(StreamingDataset):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            return result["mfcc"], result["input_ids"]


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/teamspace/s3_connections/audio-speech-hebrew", batch_size: int = BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self._already_called = {}
        for stage in ("fit", "validate", "test", "predict"):
            self._already_called[stage] = False

    def setup(self, stage: str) -> None:
        if self._already_called[stage]:
            return

        if stage == "fit":
            self.train_loader = StreamingDataLoader(SpeechStreamingDataset(input_dir=f"{self.data_dir}/train/YV"),
                                                    batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count(),
                                                     persistent_workers=True, pin_memory=True)
            self._already_called["fit"] = True
            self.val_loader = StreamingDataLoader(SpeechStreamingDataset(input_dir=f"{self.data_dir}/val"),
                                                    batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(),
                                                     persistent_workers=True, pin_memory=True)

            self._already_called["validate"] = True
            

        if stage == "validate" and not self._already_called["validate"]:
            self.val_loader = StreamingDataLoader(SpeechStreamingDataset(input_dir=f"{self.data_dir}/val"),
                                                    batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(),
                                                        persistent_workers=True, pin_memory=True)
            self._already_called["validate"] = True

        if stage == "test":
            self.test_loader = StreamingDataLoader(SpeechStreamingDataset(input_dir=f"{self.data_dir}/test"),
                                                    batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(),
                                                    persistent_workers=True, pin_memory=True)
            self._already_called["test"] = True

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader