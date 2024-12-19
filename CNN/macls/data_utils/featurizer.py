import torch
import torchaudio.compliance.kaldi as Kaldi
from torch import nn
from torchaudio.transforms import MelSpectrogram, Spectrogram, MFCC


class AudioFeaturizer(nn.Module):


    def __init__(self, feature_method='MelSpectrogram', method_args={}):
        super().__init__()
        self._method_args = method_args
        self._feature_method = feature_method
        if feature_method == 'MelSpectrogram':
            self.feat_fun = MelSpectrogram(**method_args)
        elif feature_method == 'Spectrogram':
            self.feat_fun = Spectrogram(**method_args)
        elif feature_method == 'MFCC':
            self.feat_fun = MFCC(**method_args)
        elif feature_method == 'Fbank':
            self.feat_fun = KaldiFbank(**method_args)
        else:
            raise Exception(f'Method {self._feature_method} is not exist!')

    def forward(self, waveforms, input_lens_ratio=None):
        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)
        feature = self.feat_fun(waveforms)
        feature = feature.transpose(2, 1)

        feature = feature - feature.mean(1, keepdim=True)
        if input_lens_ratio is not None:

            input_lens = (input_lens_ratio * feature.shape[1])
            mask_lens = torch.round(input_lens).long()
            mask_lens = mask_lens.unsqueeze(1)

            idxs = torch.arange(feature.shape[1], device=feature.device).repeat(feature.shape[0], 1)
            mask = idxs < mask_lens
            mask = mask.unsqueeze(-1)

            feature = torch.where(mask, feature, torch.zeros_like(feature))
        return feature

    @property
    def feature_dim(self):

        if self._feature_method == 'MelSpectrogram':
            print(self._method_args.get('n_mels', 128))
            return self._method_args.get('n_mels', 128)
        elif self._feature_method == 'Spectrogram':
            return self._method_args.get('n_fft', 400) // 2 + 1
        elif self._feature_method == 'MFCC':
            return self._method_args.get('n_mfcc', 40)
        elif self._feature_method == 'Fbank':
            return self._method_args.get('num_mel_bins', 23)
        else:
            raise Exception('No{} method'.format(self._feature_method))


class KaldiFbank(nn.Module):
    def __init__(self, **kwargs):
        super(KaldiFbank, self).__init__()
        self.kwargs = kwargs

    def forward(self, waveforms):
        log_fbanks = []
        for waveform in waveforms:
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            log_fbank = Kaldi.fbank(waveform, **self.kwargs)
            log_fbank = log_fbank.transpose(0, 1)
            log_fbanks.append(log_fbank)
        log_fbank = torch.stack(log_fbanks)
        return log_fbank
