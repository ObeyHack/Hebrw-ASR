import lightning as pl
from litdata import StreamingDataLoader, StreamingDataset
from prep.processor import get_feature_extractor, get_tokenizer, FEATURES
import os
import torch
import torchaudio.transforms as T
import torch.nn as nn
from lightning.pytorch.utilities.combined_loader import CombinedLoader

CLASSES = get_tokenizer().vocab_size + 1 # 1 for epsilon, 7 for special tokens
BATCH_SIZE = 32

class SpeechStreamingDataset(StreamingDataset):

        def __init__(self, training=False, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.training = training
            self.feature_augmentation = nn.Sequential(
                                            T.FrequencyMasking(freq_mask_param=10),
                                            T.TimeMasking(time_mask_param=50),
                                        )   

        def __getitem__(self, index):
            result = super().__getitem__(index)

            mfcc = result["mfcc"]
            mfcc_length = result["mfcc_length"].squeeze(0)
            text = result["text_encoded"]
            text_length = result["text_length"].squeeze(0)
            
            if self.training:
                mfcc1 = self.feature_augmentation(mfcc[:, :mfcc_length])
                mfcc2 = mfcc[:, mfcc_length:]
                mfcc = torch.cat([mfcc1, mfcc2], dim=1)

            return mfcc, mfcc_length, text, text_length
            # return result["mfcc"], result["mfcc_length"].squeeze(0), result["text_encoded"], result["text_length"].squeeze(0)


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/teamspace/s3_connections/audio-speech-hebrew",
                    train_dir: str = "kan_extand", 
                     batch_size: int = BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dir = train_dir
        self._already_called = {}
        for stage in ("fit", "validate", "test", "predict"):
            self._already_called[stage] = False

    def setup(self, stage: str) -> None:
        if self._already_called[stage]:
            return

        if stage == "fit":
            self.train_loader = StreamingDataLoader(
                                    SpeechStreamingDataset(
                                        training=True, 
                                        input_dir=f"{self.data_dir}/train/{self.train_dir}", 
                                        subsample=1.0
                                        ),
                                    batch_size=self.batch_size,
                                    shuffle=True, 
                                    num_workers=os.cpu_count(),
                                    persistent_workers=True, 
                                    pin_memory=True
                                    )
                                

            self._already_called["fit"] = True

            self.val_loader = StreamingDataLoader(SpeechStreamingDataset(input_dir=f"{self.data_dir}/val", 
                                                                        subsample=1.0),
                                                    batch_size=self.batch_size, 
                                                    shuffle=False, 
                                                    num_workers=os.cpu_count(),
                                                    persistent_workers=True, 
                                                    pin_memory=True)


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


def main():
    dm = AudioDataModule()
    dm.setup("fit")
    for batch in dm.train_dataloader():
        x, x_len, y, y_len = batch
        pass



if __name__ == "__main__":
    main()