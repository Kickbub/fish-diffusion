import torch
from torch import nn

from .builder import ENCODERS


@ENCODERS.register_module()
class LSTM3Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_layers=3,
    ):
        """
        LSTM encoder.

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
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, output_size)
        self.lstm = nn.LSTM(
            256,
            output_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x, *args, **kwargs):
        raw_embed = self.embedding(x)
        output, _ = self.lstm(raw_embed.unsqueeze(1))
        return output
