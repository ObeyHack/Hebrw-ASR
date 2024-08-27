import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



class BiRNN(nn.Module):
    """
    Recurrent neural network with batch normalization layer & ReLU activation function.

    Args:
        input_size (int): size of input
        hidden_state_dim (int): the number of features in the hidden state `h`
        rnn_type (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        dropout_p (float, optional): dropout probability (default: 0.1)

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs
        - **outputs**: Tensor produced by the BNReluRNN module
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,}

    def __init__(self,
            input_size: int,
            hidden_state_dim: int = 512,
            rnn_type: str = 'gru',
            bidirectional: bool = True,
            dropout: float = 0.1,):
        super(BiRNN, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.batch_norm = nn.BatchNorm1d(input_size)
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_state_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )


    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor (N, T, H)
        """

        # (N, T, H)
        x = self.batch_norm(x)

        x = F.relu(x)

        # (N, T, H)
        x = Tensor.transpose(x, 1, 2)

        # (N, H, T)
        x, _ = self.rnn(x)

        # (N, T, H)
        x = Tensor.transpose(x, 1, 2)
        return x