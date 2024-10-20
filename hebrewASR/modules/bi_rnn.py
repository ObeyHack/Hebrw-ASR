import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch


class BiRNN(nn.Module):
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,}

    def __init__(self,
            input_size: int,
            hidden_state_dim: int = 256,
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
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )


    def forward(self, x: Tensor, input_lengths: Tensor = None):
        """
        Args:
            x (Tensor): input tensor (N, H, T)
            input_lengths (Tensor): tensor containing sequence lengths (N,)
        """
        total_length = x.size(2)

        # (N, H, T)
        x = self.batch_norm(x)

        # (N, H, T)
        x = F.relu(x)

        # (N, H, T)
        x = Tensor.transpose(x, 1, 2)

        # # (N, T, H)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=input_lengths, batch_first=True, enforce_sorted=False)

        # (N, T, H)
        x, _ = self.rnn(x)

        # # (N, T, H)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=total_length, batch_first=True)

        # (N, T, H)
        x = Tensor.transpose(x, 1, 2)
        return x