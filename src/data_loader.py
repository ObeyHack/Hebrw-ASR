from tqdm import tqdm
from litdata import StreamingDataLoader, StreamingDataset


class SpeechStreamingDataset(StreamingDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        return result["mfcc"], result["input_ids"]

def main():
    dataset = SpeechStreamingDataset(input_dir="test")
    dataloader = StreamingDataLoader(dataset, batch_size=64, )

    for batch in tqdm(dataloader, total=len(dataloader)):
        x, y = batch
        print(x.shape, y.shape)


if __name__ == '__main__':
    main()