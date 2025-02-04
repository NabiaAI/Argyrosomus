import torch
import torchaudio.compliance.kaldi as Kaldi
from torch import nn
from torchaudio.transforms import MelSpectrogram, Spectrogram, MFCC
import librosa
import numpy as np
import librosa.display






class AudioFeaturizer(nn.Module):
    """音频特征器

    :param feature_method: 所使用的预处理方法
    :type feature_method: str
    :param method_args: 预处理方法的参数
    :type method_args: dict
    """

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
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')
        

    def forward(self, waveforms, input_lens_ratio=None):
        """从AudioSegment中提取音频特征

        :param waveforms: Audio segment to extract features from.
        :type waveforms: AudioSegment
        :param input_lens_ratio: input length ratio
        :type input_lens_ratio: tensor
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
                
        
        if len(waveforms.shape) == 1:
            waveforms = waveforms.unsqueeze(0)
            
            #####Spectrum 特征提取逻辑
        if self._feature_method == 'Spectrogram':
            waveforms_np = waveforms.cpu().numpy()
            features = []

            for y in waveforms_np:
                # 计算 STFT 幅值谱
                n_fft = self._method_args.get('n_fft', 256)
                hop_length = self._method_args.get('hop_length', 128)
                S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

                # 转换为 dB 幅值谱
                S_db = librosa.amplitude_to_db(S, ref=np.max)
                features.append(S_db)

            # 将结果转换为张量
            features_np = np.array(features, dtype=np.float32)
            feature = torch.tensor(features_np, dtype=torch.float32).to(waveforms.device)
        else:
        
            feature = self.feat_fun(waveforms)
        
           ## 如果是 MelSpectrogram，只保留前一半特征
        # if self._feature_method == 'MelSpectrogram':
        #     feature = feature[:,:feature.shape[1] // 2, :]
        

        
        feature = feature.transpose(2, 1)

        # 归一化
        feature = feature - feature.mean(1, keepdim=True)
        # 我修改，归一化：减去均值并除以标准差 ###对每一个音频进行归一化
        # feature_mean = feature.mean(1, keepdim=True)
        # feature_std = feature.std(1, keepdim=True) + 1e-8  # 防止除以零
        # feature = (feature - feature_mean) / feature_std
        
        ############对每个batch进行归一化
        # feature_mean = feature.mean(dim=(0, 1), keepdim=True)  # 针对 batch 计算均值
        # feature_std = feature.std(dim=(0, 1), keepdim=True) + 1e-8  # 针对 batch 计算标准差
        # feature = (feature - feature_mean) / feature_std
        

        # feature = normalize_frequency_bands(feature,num_bands=4)
        #feature = normalize_frequency_bands_sp(feature, n_fft, self._method_args.get('sample_rate', 4000), num_bands=5)
        
        mask = None


        if input_lens_ratio is not None:
            # 对掩码比例进行扩展
            input_lens = (input_lens_ratio * feature.shape[1])
            mask_lens = torch.round(input_lens).long()
            mask_lens = mask_lens.unsqueeze(1)
            # 生成掩码张量
            idxs = torch.arange(feature.shape[1], device=feature.device).repeat(feature.shape[0], 1)
            mask = idxs < mask_lens
            ############# 12.31修改
            mask = mask.unsqueeze(-1)
            # 对特征进行掩码操作
            #feature = torch.where(mask, feature, torch.zeros_like(feature))

        return feature

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self._feature_method == 'MelSpectrogram':
            #print(self._method_args.get('n_mels', 128))
            return self._method_args.get('n_mels', 128)
            #return self._method_args.get('n_mels', 128) // 2
        elif self._feature_method == 'Spectrogram':
            return self._method_args.get('n_fft', 256) // 2 + 1
        elif self._feature_method == 'MFCC':
            return self._method_args.get('n_mfcc', 40)
        elif self._feature_method == 'Fbank':
            return self._method_args.get('num_mel_bins', 23)
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))
        


class KaldiFbank(nn.Module):
    def __init__(self, **kwargs):
        super(KaldiFbank, self).__init__()
        self.kwargs = kwargs

    def forward(self, waveforms):
        """
        :param waveforms: [Batch, Length]
        :return: [Batch, Length, Feature]
        """
        log_fbanks = []
        for waveform in waveforms:
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            log_fbank = Kaldi.fbank(waveform, **self.kwargs)
            log_fbank = log_fbank.transpose(0, 1)
            log_fbanks.append(log_fbank)
        log_fbank = torch.stack(log_fbanks)
        return log_fbank
    
 
import torch

def normalize_frequency_bands(spectrogram, num_bands=3):
    """
    对频谱进行分频带归一化（支持 PyTorch Tensor）。
    :param spectrogram: 输入频谱，形状为 (Batch, Time, Frequency)。
    :param num_bands: 划分的频带数量。
    :return: 归一化后的频谱，形状与输入相同。
    """
    # 获取频率维度
    batch_size, num_time_steps, num_freq_bins = spectrogram.size()
    band_size = num_freq_bins // num_bands

    normalized_spectrogram = spectrogram.clone()

    for i in range(num_bands):
        start_freq = i * band_size
        end_freq = (i + 1) * band_size if i != num_bands - 1 else num_freq_bins
        band = spectrogram[:, :, start_freq:end_freq]

        # 计算均值和标准差（沿频率轴计算）
        mean = band.mean(dim=2, keepdim=True)  # 维度 [Batch, Time, 1]
        std = band.std(dim=2, keepdim=True)  # 维度 [Batch, Time, 1]

        # 避免分母为零
        std += 1e-6

        # 归一化
        normalized_band = (band - mean) / std
        normalized_spectrogram[:, :, start_freq:end_freq] = normalized_band

    return normalized_spectrogram


import numpy as np
import librosa

def get_frequency_bins(n_fft, sample_rate):
    # 计算每个频率 bin 对应的频率
    freqs = np.fft.rfftfreq(n_fft, d=1/sample_rate)
    return freqs

def split_spectrogram_by_frequency(spectrogram, sample_rate, n_fft, low_freq=100, high_freq=1000):
    # 获取频率 bin 对应的频率
    freqs = get_frequency_bins(n_fft, sample_rate)

    # 找到频率范围对应的 bin 索引
    low_idx = np.where(freqs >= low_freq)[0][0]
    high_idx = np.where(freqs <= high_freq)[0][-1]

    # 分离频谱
    low_band = spectrogram[:low_idx, :]
    high_band = spectrogram[high_idx:, :]

    return low_band, high_band

def normalize_frequency_bands_sp(spectrogram, n_fft, sample_rate, num_bands=5):
    freqs = get_frequency_bins(n_fft, sample_rate)
    band_size = len(freqs) // num_bands  # 每个频带的 bin 数量

    normalized_spectrogram = spectrogram.clone()
    for i in range(num_bands):
        start_idx = i * band_size
        end_idx = (i + 1) * band_size if i < num_bands - 1 else len(freqs)
        band = spectrogram[start_idx:end_idx, :]
        band_mean = band.mean(axis=2, keepdims=True)
        band_std = band.std(axis=2, keepdims=True) + 1e-8
        normalized_spectrogram[start_idx:end_idx, :] = (band - band_mean) / band_std

    return normalized_spectrogram
