import numpy as np
from .performousv3 import *

from .builder import PITCH_EXTRACTORS, BasePitchExtractor

@PITCH_EXTRACTORS.register_module()
class PerfPitchExtractor(BasePitchExtractor):
    def __call__(self, x, sampling_rate=44100, pad_to=None):

        """Extract pitch using perfv3.

        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """

        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

        # Extract pitch using libf0.salience
        f0 = bestpitch(
            y=x[0].cpu().numpy(),
            sr=sampling_rate,
            hop_length=128,
            n_fft=512,
        )

        return self.post_process(x, sampling_rate, f0, pad_to)