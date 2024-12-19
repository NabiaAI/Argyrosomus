import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from macls.data_utils.audio import AudioSegment
from macls.data_utils.featurizer import AudioFeaturizer
from macls.utils.logger import setup_logger

logger = setup_logger(__name__)


class MAClsDataset(torch.utils.data.Dataset):
    def __init__(self, data_list_path, audio_featurizer, do_vad=True, max_duration=3, min_duration=0.5, mode='train',
                 sample_rate=16000, aug_conf={}, num_speakers=1000, use_dB_normalization=True, target_dB=-20):
        super(MAClsDataset, self).__init__()
        assert mode in ['train', 'eval', 'create_data', 'extract_feature']
        self.do_vad = do_vad
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.mode = mode
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self.aug_conf = aug_conf
        self.num_speakers = num_speakers
        self.noises_path = None
        self.audio_featurizer = audio_featurizer
        self.max_feature_len = self.get_crop_feature_len()

        with open(data_list_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __getitem__(self, idx):

        data_path, label_str = self.lines[idx].replace('\n', '').split('\t')
        labels = list(map(int, label_str.split()))  

        if data_path.endswith('.npy'):
            feature = np.load(data_path)
            if feature.shape[0] > self.max_feature_len:
                crop_start = random.randint(0, feature.shape[0] - self.max_feature_len) if self.mode == 'eval' else 0
                feature = feature[crop_start:crop_start + self.max_feature_len, :]
            feature = torch.tensor(feature, dtype=torch.float32)
        else:
            audio_path, label_str = self.lines[idx].strip().split('\t')
            labels = list(map(int, label_str.split()))  


            audio_segment = AudioSegment.from_file(audio_path)

            if self.do_vad:
                audio_segment.vad()

            if self.mode == 'train':
                if audio_segment.duration < self.min_duration:
                    return self.__getitem__(idx + 1 if idx < len(self.lines) - 1 else 0)

            if audio_segment.sample_rate != self._target_sample_rate:
                audio_segment.resample(self._target_sample_rate)

            if self.mode == 'train':
                audio_segment = self.augment_audio(audio_segment, **self.aug_conf)

            if self._use_dB_normalization:
                audio_segment.normalize(target_db=self._target_dB)

            if self.mode != 'extract_feature' and audio_segment.duration > self.max_duration:
                audio_segment.crop(duration=self.max_duration, mode=self.mode)
            samples = torch.tensor(audio_segment.samples, dtype=torch.float32)
            feature = self.audio_featurizer(samples)
            feature = feature.squeeze(0)
        labels = torch.tensor(labels, dtype=torch.int32)  
        return feature, labels

    def __len__(self):
        return len(self.lines)

    def get_crop_feature_len(self):
        samples = torch.randn((1, self.max_duration * self._target_sample_rate))
        feature = self.audio_featurizer(samples).squeeze(0)
        freq_len = feature.size(0)
        return freq_len


    def augment_audio(self, audio_segment, speed_perturb=False, volume_perturb=False, volume_aug_prob=0.2, noise_dir=None,
                      noise_aug_prob=0.2):

        if speed_perturb:
            speeds = [1.0, 0.9, 1.1]
            speed_idx = random.randint(0, 2)
            speed_rate = speeds[speed_idx]
            if speed_rate != 1.0:
                audio_segment.change_speed(speed_rate)

        if volume_perturb and random.random() < volume_aug_prob:
            min_gain_dBFS, max_gain_dBFS = -15, 15
            gain = random.uniform(min_gain_dBFS, max_gain_dBFS)
            audio_segment.gain_db(gain)

        if self.noises_path is None and noise_dir is not None:
            self.noises_path = []
            if noise_dir is not None and os.path.exists(noise_dir):
                for file in os.listdir(noise_dir):
                    self.noises_path.append(os.path.join(noise_dir, file))

        if len(self.noises_path) > 0 and random.random() < noise_aug_prob:
            min_snr_dB, max_snr_dB = 10, 50
            noise_path = random.sample(self.noises_path, 1)[0]
            noise_segment = AudioSegment.slice_from_file(noise_path)
            if noise_segment.sample_rate != audio_segment.sample_rate:
                noise_segment.resample(audio_segment.sample_rate)

            snr_dB = random.uniform(min_snr_dB, max_snr_dB)

            if noise_segment.duration < audio_segment.duration:
                diff_duration = audio_segment.num_samples - noise_segment.num_samples
                noise_segment._samples = np.pad(noise_segment.samples, (0, diff_duration), 'wrap')
            audio_segment.add_noise(noise_segment, snr_dB)
        return audio_segment

