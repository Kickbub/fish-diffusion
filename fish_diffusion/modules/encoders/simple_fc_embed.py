import torch
from torch import nn

from .builder import ENCODERS


@ENCODERS.register_module()
class FCEmbedding(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
    ):
        """
        Embedding + FC (2) layer encoder.

        Args:
            input_size (int): Input size.
            output_size (int): Output size.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.embedding_fc = nn.Sequential(
            nn.Embedding(input_size, output_size),
            nn.Linear(output_size, output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size),
            nn.GELU(),
        )

    def forward(self, x, *args, **kwargs):
        concat_outputs = ()
        for id in x:
            outputs = self.embedding_fc(id)
            concat_outputs = concat_outputs + (outputs,)
        return torch.stack(concat_outputs, dim=0)
