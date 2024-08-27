import torch.nn as nn


class CNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            in_channels: int = 1,
            out_channels: int = 16,
            activation: str = 'Hardtanh',) -> None:
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.activation = nn.Hardtanh() if activation == 'Hardtanh' else nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(1, 2), padding=(20, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation)
        self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(1, 2), padding=(10, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation,)


    
    def forward(self, inputs):
        """
        inputs: torch.FloatTensor (batch, time, dimension)
        input_lengths: torch.IntTensor (batch)
        """
        intput = inputs.unsqueeze(1).transpose(2, 3)

        outputs = self.conv1(intput)

        outputs = self.conv2(outputs)

        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.contiguous().view(batch_size, seq_lengths * channels, dimension)

        return outputs

    
    def  get_output_dim(self):
        return (self.input_dim // 4) * self.out_channels