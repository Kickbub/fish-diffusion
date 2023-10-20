import torch
from torch import nn

from .builder import ENCODERS


@ENCODERS.register_module()
class LSTM2Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_layers=2,
    ):
        """
        2 layer LSTM encoder.

        Args:
            input_size (int): Input size.
            output_size (int): Output size.
            num_layers (int, optional): Number of LSTM layers. Defaults to 3.
            bidirectional (bool, optional): Whether to use bidirectional LSTM. Defaults to False.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, output_size)
        self.lstm = nn.LSTM(
            output_size,
            output_size,
            num_layers=num_layers,
            dropout=0.5,
            batch_first=True,
        )
        self.linear = nn.Linear(output_size, output_size)

    def forward(self, x, *args, **kwargs):
        concat_output = ()
        for id in x:
            raw_embed = self.embedding(id)
            _, (hidden, _) = self.lstm(raw_embed[None, None, :])
            output = torch.relu(self.linear(hidden[-1]))
            concat_output = concat_output + (output, )
        return torch.stack(concat_output)
