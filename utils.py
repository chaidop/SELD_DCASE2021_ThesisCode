
"""
using frequency masking for the ensemble method
"""
import numpy as np
import tensorflow as tf
import shape_util

class EnsembleMaskingNumpyBase:
    """
    Base class for data augmentation for audio spectrogram of numpy array. This class does not alter label
    """
    def __init__(self, always_apply: bool = True, p: float = 0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, x: np.ndarray):
        if self.always_apply:
            return self.apply(x)
        else:
            if np.random.rand() < self.p:
                return self.apply(x), True
            else:
                return x, False

    def apply(self, x: np.ndarray):
        raise NotImplementedError

class EnsembleFreqMasking(EnsembleMaskingNumpyBase):
    """
    This data augmentation randomly remove horizontal or vertical strips from image. Tested
    Time and Frequency masking
    """
    def __init__(self, always_apply: bool = False, p: float = 0.5, 
                 freq_max_width: int = 64, n_freq_stripes: int = 1,
                 n_zero_channels: int = None, is_filled_last_channels: bool = True, min_freq: int = 0, max_freq: int = 64):
        """
        :param always_apply: If True, always apply transform.
        :param p: If always_apply is false, p is the probability to apply transform.
        :param freq_max_width: maximum freq width to remove.
        :param n_freq_stripes: number of freq stripes to remove.
        :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
        :param is_filled_last_channels: if False, does not cutout n_zero_channels
        """
        super().__init__(always_apply, p)
        self.freq_max_width = freq_max_width
        self.n_freq_stripes = n_freq_stripes
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

        self.min_freq = min_freq
        self.max_freq = max_freq

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: <(n_channels, n_time_steps, n_features)>: input spectrogram.
        :return: augmented spectrogram.
        """
        assert x.ndim == 4, 'Error: dimension of input spectrogram is not 3!'
        n_frames = x.shape[2]
        n_freqs = x.shape[3]
        min_value = self.min_freq
        max_value = self.max_freq

        freq_max_width = self.freq_max_width
        freq_max_width = np.max((1, freq_max_width))

        new_spec = x.copy()

        for i in np.arange(self.n_freq_stripes):
            dur = max_value[i] - min_value[i] ## diastima apokopis
            #gia to 1o model: 0-12kHz or 0-32 mel-bins
            #gia to 2o model: 6-18 kHZ or 16-48 mel-bins
            #gia ti 3o model: 12-24 kHz or 32-64 mel-bins
            start_idx = min_value[i]
            new_spec[:,:, :, start_idx:start_idx + dur] = 0.0

        return new_spec