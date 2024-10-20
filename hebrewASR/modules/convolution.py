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
                nn.Conv2d(in_channels, out_channels, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation)
        self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5), bias=False),
                nn.BatchNorm2d(out_channels),
                self.activation,)


    
    def forward(self, inputs, inputs_length):
        """
        inputs: torch.FloatTensor (N, F, T)
        inputs_length: torch.IntTensor (N)
        
        return: torch.FloatTensor (N, F * out_channels, T)
                torch.IntTensor (N)

        """
        inputs = inputs.unsqueeze(1)

        outputs = self.conv1(inputs)

        outputs = self.conv2(outputs)

        N, channels, F, T = outputs.size()
        
        outputs = outputs.view(N, F * channels, T)

        return outputs, self._get_sequence(inputs_length)

    
    def  get_output_dim(self):
        return (self.input_dim // 4) * self.out_channels


    def _get_sequence_length_model(self, module: nn.Module, seq_lengths):
        """
        Calculate convolutional neural network receptive formula
        Args:
            module (torch.nn.Module): module of CNN
            seq_lengths (torch.IntTensor): The actual length of each sequence in the batch
        Returns: seq_lengths
            - **seq_lengths**: Sequence length of output from the module
        """
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[1])
            seq_lengths = seq_lengths.int() + 1

        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1

        return seq_lengths.int()

    
    def _get_sequence(self, seq_lengths):
        """
        Calculate the output sequence length of the CNN
        Args:
            seq_lengths (torch.IntTensor): The actual length of each sequence in the batch
        Returns: seq_lengths
            - **seq_lengths**: Sequence length of output from the module
        """
        seq_lengths = self._get_sequence_length_model(self.conv1[0], seq_lengths)
        seq_lengths = self._get_sequence_length_model(self.conv2[0], seq_lengths)
        return seq_lengths
