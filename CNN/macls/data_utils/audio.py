import copy
import io
import os
import random

import numpy as np
import resampy
import soundfile

from macls.data_utils.utils import buf_to_float, vad, decode_audio


class AudioSegment(object):

    def __init__(self, samples, sample_rate):
        self._samples = self._convert_samples_to_float32(samples)
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (f"{type(self)}: num_samples={self.num_samples}, sample_rate={self.sample_rate}, "
                f"duration={self.duration:.2f}sec, rms={self.rms_db:.2f}dB")

    @classmethod
    def from_file(cls, file):
        assert os.path.exists(file), f'file not exist{file}'
        try:
            samples, sample_rate = soundfile.read(file, dtype='float32')
        except:
            sample_rate = 16000
            samples = decode_audio(file=file, sample_rate=sample_rate)
        return cls(samples, sample_rate)

    @classmethod
    def slice_from_file(cls, file, start=None, end=None):
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = 0. if start is None else round(start, 3)
        end = duration if end is None else round(end, 3)
  
        if start < 0.0: start += duration
        if end < 0.0: end += duration
  
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError(f"End({end} s)out of range")
        if start > end:
            raise ValueError(f"Start({start} s)out of range({end} s)")
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        data = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return cls(data, sample_rate)

    @classmethod
    def from_bytes(cls, data):

        samples, sample_rate = soundfile.read(io.BytesIO(data), dtype='float32')
        return cls(samples, sample_rate)

    @classmethod
    def from_pcm_bytes(cls, data, channels=1, samp_width=2, sample_rate=16000):

        samples = buf_to_float(data, n_bytes=samp_width)
        if channels > 1:
            samples = samples.reshape(-1, channels)
        return cls(samples, sample_rate)

    @classmethod
    def from_ndarray(cls, data, sample_rate=16000):

        return cls(data, sample_rate)

    @classmethod
    def concatenate(cls, *segments):

        # Perform basic sanity-checks.
        if len(segments) == 0:
            raise ValueError("No segmenets are not concatenated")
        sample_rate = segments[0]._sample_rate
        for seg in segments:
            if sample_rate != seg._sample_rate:
                raise ValueError("Use different sample rate")
            if type(seg) is not cls:
                raise TypeError("Only same type can be concatenated")
        samples = np.concatenate([seg.samples for seg in segments])
        return cls(samples, sample_rate)

    @classmethod
    def make_silence(cls, duration, sample_rate):

        samples = np.zeros(int(duration * sample_rate))
        return cls(samples, sample_rate)

    def to_wav_file(self, filepath, dtype='float32'):

        samples = self._convert_samples_from_float32(self._samples, dtype)
        subtype_map = {
            'int16': 'PCM_16',
            'int32': 'PCM_32',
            'float32': 'FLOAT',
            'float64': 'DOUBLE'
        }
        soundfile.write(
            filepath,
            samples,
            self._sample_rate,
            format='WAV',
            subtype=subtype_map[dtype])

    def superimpose(self, other):

        if not isinstance(other, type(self)):
            raise TypeError(f"Can not add: {type(self)} and {type(other)}")
        if self._sample_rate != other._sample_rate:
            raise ValueError("Must use the same sample rate")
        if len(self._samples) != len(other._samples):
            raise ValueError("Length should be match")
        self._samples += other._samples

    def to_bytes(self, dtype='float32'):

        samples = self._convert_samples_from_float32(self._samples, dtype)
        return samples.tostring()

    def to(self, dtype='int16'):

        samples = self._convert_samples_from_float32(self._samples, dtype)
        return samples

    def gain_db(self, gain):

        self._samples *= 10. ** (gain / 20.)

    def change_speed(self, speed_rate):

        if speed_rate == 1.0:
            return
        if speed_rate <= 0:
            raise ValueError("speed_rate should larger tha 0")
        old_length = self._samples.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        self._samples = np.interp(new_indices, old_indices, self._samples).astype(np.float32)

    def normalize(self, target_db=-20, max_gain_db=300.0):

        gain = target_db - self.rms_db
        if gain > max_gain_db:
            raise ValueError(f"Cannot normalized{target_db}dB，audio gain{gain}gain is larger than max_gain_db ({max_gain_db}dB)")
        self.gain_db(min(max_gain_db, target_db - self.rms_db))

    def resample(self, target_sample_rate, filter='kaiser_best'):

        self._samples = resampy.resample(self.samples, self.sample_rate, target_sample_rate, filter=filter)
        self._sample_rate = target_sample_rate

    def pad_silence(self, duration, sides='both'):

        if duration == 0.0:
            return self
        cls = type(self)
        silence = self.make_silence(duration, self._sample_rate)
        if sides == "beginning":
            padded = cls.concatenate(silence, self)
        elif sides == "end":
            padded = cls.concatenate(self, silence)
        elif sides == "both":
            padded = cls.concatenate(silence, self, silence)
        else:
            raise ValueError(f"Unknown value for the sides {sides}")
        self._samples = padded._samples

    def shift(self, shift_ms):

        if abs(shift_ms) / 1000.0 > self.duration:
            raise ValueError("shift_ms should less than the duration of audio")
        shift_samples = int(shift_ms * self._sample_rate / 1000)
        if shift_samples > 0:
            # time advance
            self._samples[:-shift_samples] = self._samples[shift_samples:]
            self._samples[-shift_samples:] = 0
        elif shift_samples < 0:
            # time delay
            self._samples[-shift_samples:] = self._samples[:shift_samples]
            self._samples[:-shift_samples] = 0

    def subsegment(self, start_sec=None, end_sec=None):

        start_sec = 0.0 if start_sec is None else start_sec
        end_sec = self.duration if end_sec is None else end_sec
        if start_sec < 0.0:
            start_sec = self.duration + start_sec
        if end_sec < 0.0:
            end_sec = self.duration + end_sec
        if start_sec < 0.0:
            raise ValueError(f"Start({start_sec} s)out of range")
        if end_sec < 0.0:
            raise ValueError(f"End ({end_sec} s)out of range")
        if start_sec > end_sec:
            raise ValueError(f"Start({start_sec} s)later than({end_sec} s)")
        if end_sec > self.duration:
            raise ValueError(f"End({end_sec} s)out of range(> {self.duration} s)")
        start_sample = int(round(start_sec * self._sample_rate))
        end_sample = int(round(end_sec * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]

    def random_subsegment(self, subsegment_length):

        if subsegment_length > self.duration:
            raise ValueError("Length of subsegment must not be greater "
                             "than original segment.")
        start_time = random.uniform(0.0, self.duration - subsegment_length)
        self.subsegment(start_time, start_time + subsegment_length)

    def add_noise(self,
                  noise,
                  snr_dB,
                  max_gain_db=300.0):

        if noise.sample_rate != self.sample_rate:
            raise ValueError(f"Noise sample rate({noise.sample_rate} Hz)is not equal to base SR({self.sample_rate} Hz)")
        if noise.duration < self.duration:
            raise ValueError(f"The duration of noise({noise.duration}second)must at least as equal as({self.duration}second)")
        noise_gain_db = min(self.rms_db - noise.rms_db - snr_dB, max_gain_db)
        noise_new = copy.deepcopy(noise)
        noise_new.random_subsegment(self.duration)
        noise_new.gain_db(noise_gain_db)
        self.superimpose(noise_new)

    def vad(self, top_db=20, overlap=200):
        self._samples = vad(wav=self._samples, top_db=top_db, overlap=overlap)

    # 裁剪音频
    def crop(self, duration, mode='eval'):
        if self.duration > duration:
            if mode == 'train':
                self.random_subsegment(duration)
            else:
                self.subsegment(end_sec=duration)

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_samples(self):
        return self._samples.shape[0]

    @property
    def duration(self):
        return self._samples.shape[0] / float(self._sample_rate)

    @property
    def rms_db(self):
        # square root => multiply by 10 instead of 20 for dBs
        mean_square = np.mean(self._samples ** 2)
        if mean_square == 0:
            mean_square = 1
        return 10 * np.log10(mean_square)

    @staticmethod
    def _convert_samples_to_float32(samples):
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= (1. / 2 ** (bits - 1))
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError(f"Unsupported sample type: {samples.dtype}.")
        return float32_samples

    @staticmethod
    def _convert_samples_from_float32(samples, dtype):
        dtype = np.dtype(dtype)
        output_samples = samples.copy()
        if dtype in np.sctypes['int']:
            bits = np.iinfo(dtype).bits
            output_samples *= (2 ** (bits - 1) / 1.)
            min_val = np.iinfo(dtype).min
            max_val = np.iinfo(dtype).max
            output_samples[output_samples > max_val] = max_val
            output_samples[output_samples < min_val] = min_val
        elif samples.dtype in np.sctypes['float']:
            min_val = np.finfo(dtype).min
            max_val = np.finfo(dtype).max
            output_samples[output_samples > max_val] = max_val
            output_samples[output_samples < min_val] = min_val
        else:
            raise TypeError(f"Unsupported sample type: {samples.dtype}.")
        return output_samples.astype(dtype)
