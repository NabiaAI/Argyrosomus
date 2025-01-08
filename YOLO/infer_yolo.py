import numpy as np
import sys
import math
import os
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.io import wavfile as wav
import time
import json
import pandas as pd

sys.path.append('.')
from create_data_yolo import save_spectrogram, read_audio_file, normalize_audio

def y_coord_to_freq(coord):
    # coord = freq / (sr / n_fft) # sr / n_fft = is size of each bin
    # => freq = coord * (sr / n_fft)
    sr, n_fft = 4000, 256 # MUST BE EXACTLY THE SAME AS FOR THE SPECTROGRAM CREATION
    return coord * (sr / n_fft)

def x_coord_to_time(coord):
    # coord = coord / (sr / hop_length) # sr / hop_length = number of bins
    sr, hop_length = 4000, 64 # MUST BE EXACTLY THE SAME AS FOR THE SPECTROGRAM CREATION
    return coord / (sr / hop_length) # coord * (sr / hop_length)
    

def convert_to_raven_selection_table(input_array, time_shift=0, img_height=64):
    img_height = 64 # RESULTS FROM SPECTROGRAM CREATION 
    data = []
    for idx, row in enumerate(input_array):
        x1, y1, x2, y2, category, conf = row
        x1, y1, x2, y2, conf = map(float, [x1, y1, x2, y2, conf]) # Convert strings to floats

        # Calculate the desired output columns
        begin_time = x_coord_to_time(x1) + time_shift
        end_time = x_coord_to_time(x2) + time_shift
        low_freq = y_coord_to_freq(img_height - y2) # y=0 is at top of spectrogram -> invert
        high_freq = y_coord_to_freq(img_height- y1)

        data.append([
            idx + 1,  # Selection starts from 1
            "Spectrogram 1",  # View
            1,  # Channel
            begin_time,
            end_time,
            low_freq,
            high_freq,
            category,
            conf
        ])

    columns = [
        "Selection", "View", "Channel", "Begin Time (s)", "End Time (s)",
        "Low Freq (Hz)", "High Freq (Hz)", "category", "confidence"
    ]
    return pd.DataFrame(data, columns=columns)

def extract_time_stamp(file_path:str):
    if file_path.endswith('_.wav'):
        idx = file_path.find('_.wav')
        hour_str = file_path[idx-4:idx] # 0416_.wav => 4h 16 min
    else:
        hour_str = file_path.split('_')[-1].split('.')[0] # eg date_120000.wav => 120000 = 12h; 
    assert hour_str.isdigit() and (len(hour_str) == 6 or len(hour_str) == 4), f"Hour string is not in the correct format: {hour_str}"
    hour = int(hour_str[:2])
    minute = int(hour_str[2:4])
    time_in_seconds = hour * 3600 + minute * 60
    return time_in_seconds

def segment_audios(file_paths, segment_duration=5, stride=None, extract_timestamps=True):
    if stride is None:
        stride = segment_duration

    time_stamps = []
    all_segments = []
    for file_path in file_paths:
        base_time_of_file = extract_time_stamp(file_path) if extract_timestamps else 0
        sample_rate, audio_data  = wav.read(file_path)
        
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        segment_samples = segment_duration * sample_rate
        stride_samples = stride * sample_rate

        for i in range(0, len(audio_data) - segment_samples + 1, stride_samples):
            segment = audio_data[i:i + segment_samples]
            all_segments.append(segment)
            time_stamps.append(base_time_of_file + i / sample_rate)

    return time_stamps, all_segments, sample_rate

def detection_to_cls_results(preds, thresholds=None):
    cls_map = {0: 2, 1:1, 2:0} # other way round in cls labels
    converted_results = []
    converted_probs = []
    for pred in preds:
        one_hot = np.zeros((len(pred.names),))
        prob = np.zeros((len(pred.names),))
        for b in pred.boxes:
            cls = int(b.cls[0])
            conf = b.conf[0]
            one_hot[cls_map[cls]] = 1
            if conf > prob[cls_map[cls]]: # take the highest confidence (better to optimize threshold)
                prob[cls_map[cls]] = conf
        converted_results.append(one_hot)
        converted_probs.append(prob)
    converted_results = np.array(converted_results)
    converted_probs = np.array(converted_probs)
    if thresholds is not None:
        converted_results = (converted_probs > np.array(thresholds)).astype('int')
    return converted_results, converted_probs

def filter_too_short(audios, labels, srs, min_duration=0.1):
    without_labels = False
    if labels is None:
        without_labels = True
        labels = [None] * len(audios)
    filtered_audios = []
    filtered_labels = []
    filtered_srs = []
    for audio, label, sr in zip(audios, labels, srs, strict=True):
        if len(audio) >= min_duration * sr:
            filtered_audios.append(audio)
            filtered_labels.append(label)
            filtered_srs.append(sr)
    if without_labels:
        return filtered_audios, None, filtered_srs
    return filtered_audios, filtered_labels, filtered_srs

def numpy_audio_to_spectrogram(audio, sr, target_duration=5):
    if len(audio) < target_duration * sr:
        length = int(target_duration * sr)
        audio = np.pad(audio, (0, length - len(audio)), 'constant')
    if len(audio) > target_duration * sr:
        print(f"WARNING: Audio is longer than target duration -> CROPPING (cut {len(audio) / sr - target_duration:.3}s off)")
        audio = audio[:int(target_duration * sr)]
    spectrogram = save_spectrogram(audio, sr, as_array=True)
    return spectrogram


def load_list(input):
    assert os.path.exists(input), f"Input path {input} does not exist"
    if os.path.isfile(input):
        with open(input) as f:
            lines = f.readlines()
    else: # dir
        lines = []
        for root, _, files in os.walk(input):
            for file in files:
                if file.lower().endswith('.wav'):
                    lines.append(os.path.join(root, file))

    labels_list = []
    audio_list = []
    srs = []
    skipped = 0
    for line in lines:
        results = line.strip().split('\t')
        file_path = results[0]
        if len(results) == 2:
            labels = results[1]
            labels = labels.split(' ')
        else:
            labels = [-1]
        if not os.path.exists(file_path):
            skipped += 1
            continue
        sr, audio = read_audio_file(file_path)
        labels_list.append(labels)
        audio_list.append(audio)
        srs.append(sr)

    if skipped > 0:
        print(f"WARNING: Skipped {skipped} files which did not exist")
    return audio_list, labels_list, srs

def audios_to_spectrograms(audios: list[np.array], srs, target_duration, executor=None): # TODO: split audio if its too long
    spectrogram_list = []
    idxs = []

    close_executor = False
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=os.cpu_count()-1)
        close_executor = True

    futures = {executor.submit(numpy_audio_to_spectrogram, audio, sr, target_duration): idx for idx, (audio, sr) in enumerate(zip(audios, srs, strict=True))}
    for future in tqdm(as_completed(futures), total=len(futures)):
        idx = futures[future]
        spectrogram = future.result()
        spectrogram_list.append(spectrogram)
        idxs.append(idx)

    if close_executor:
        executor.shutdown()

    idx = np.argsort(np.array(idxs)) # sort back to original order
    spectrogram_list = np.array(spectrogram_list)[idx]
    return spectrogram_list

def load_cached(input, cache_path=None, min_duration=0.1, target_duration=5, no_labels=False, sr=None, executor=None):
    if cache_path:
        try:
            if no_labels:
                return np.load(f"{cache_path}/spectrograms.npy")
            return np.load(f"{cache_path}/spectrograms.npy"), np.load(f"{cache_path}/labels.npy")
        except FileNotFoundError:
            pass

    if isinstance(input, str):
        audio_list, labels, srs = load_list(input)
    else:
        assert sr is not None, "sr must be provided if input is not a file path"
        assert isinstance(input, list), "input must be a list of numpy arrays"
        audio_list = input
        srs = [sr] * len(input)
        labels = None

    audio_list, labels, srs = filter_too_short(audio_list, labels, srs, min_duration)
    labels = np.array(labels, dtype='int') if labels is not None else None
    spectrogram_list = audios_to_spectrograms(audio_list, srs, target_duration, executor=executor)

    if cache_path:
        os.makedirs(cache_path, exist_ok=True)
        np.save(f"{cache_path}/spectrograms.npy", spectrogram_list)
        if not no_labels:
            np.save(f"{cache_path}/labels.npy", labels)
    if no_labels:
        return spectrogram_list
    return spectrogram_list, labels

def get_boxes(results):
    boxes = []
    for result in results:
        cls = np.array([result.names[c] for c in result.boxes.cls])
        box_list = np.hstack([result.boxes.xyxy, cls.reshape((-1, 1)), result.boxes.conf.reshape((-1, 1))])
        boxes.append(box_list)
    return boxes

class YOLOMultiLabelClassifier:
    def __init__(self, model_path, *, device='mps', bounding_box_threshold=0.25, thresholds=None):
        from ultralytics import YOLO
        if isinstance(model_path, str):
            self._model = YOLO(f"{model_path}/best.pt")
        else:
            self._model = model_path
        self.bounding_box_threshold = bounding_box_threshold
        self.device = device
        self.imgsz = (64,320) # based on 4khz audio with 5s duration up to 1000 Hz

        self.thresholds = None
        if thresholds and isinstance(thresholds, str):
            self.thresholds = self.evaluate_and_adjust_thresholds(thresholds, ratio=1.0)
        elif thresholds:
            self.thresholds = thresholds
        else:
            with open(f"{model_path}/thresholds.json") as f:
                self.thresholds = json.load(f)
            # self.thresholds =  #[0.2650397717952728, 0.5434091091156006, 0.3506118953227997]

    def evaluate_and_adjust_thresholds(self, threshold_train_path, ratio):
        spectrograms, labels = load_cached(threshold_train_path, cache_path=None)
        idx = np.random.choice(len(spectrograms), int(ratio * len(spectrograms)), replace=False)
        spectrograms, labels = spectrograms[idx], labels[idx]
        _, probs = self.predict(spectrograms)
        optimal_thresholds = []
        optimal_precisions = []
        optimal_recalls = []
        for i in range(labels.shape[1]):
            precision, recall, thresholds = precision_recall_curve(labels[:, i], probs[:, i])
            f1_scores = 2 * precision * recall / (precision + recall)
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            optimal_thresholds.append(float(optimal_threshold))
            optimal_precisions.append(float(precision[np.argmax(f1_scores)]))
            optimal_recalls.append(float(recall[np.argmax(f1_scores)]))
        print("Optimal thresholds for each label:\t\t", optimal_thresholds)
        print("Producing precisions for each label:\t\t", optimal_precisions)
        print("Producing recalls for each label:\t\t", optimal_recalls)
        return optimal_thresholds
    
    def predict_file(self, file_path, *, save=False, raven_table=False):
        segment_duration = 5
        _, audios, sr = segment_audios([file_path], segment_duration=segment_duration, extract_timestamps=False)
        audios = [normalize_audio(audio) for audio in audios]
        spectrogram = load_cached(audios, cache_path=None, sr=sr, no_labels=True)
        preds, probs, boxes = self.predict(spectrogram, save=save, return_boxes=True)

        if not raven_table:
            return preds, probs, boxes

        time_shift = 0
        frames = []
        for b in boxes:
            df = convert_to_raven_selection_table(b, time_shift, img_height=spectrogram[0].shape[1])
            if len(df) == 0:
                time_shift += segment_duration
                continue
            df.set_index("Selection", inplace=True)
            frames.append(df)
            time_shift += segment_duration

        df = pd.concat(frames, ignore_index=True)
        df.reset_index(inplace=True, names="Selection")
        df["Selection"] += 1 # start from 1
        df.to_csv(f"{os.path.splitext(file_path)[0]}.Table.1.selections.predicted.txt", sep="\t", index=False)
        return preds, probs, boxes

    def predict(self, input, *, save=False, batch_size=64, return_boxes=False):
        if isinstance(input, np.ndarray) and input.ndim > 3 or isinstance(input, list) and isinstance(input[0], np.ndarray):
            input = [spectro[...,::-1] for spectro in input]
        
        # transorm into tensors of batch size
        n_batches = math.ceil(len(input) / batch_size)
        predictions = []
        probabilities = []
        boxes = []
        for i in range(n_batches):
            batch = input[i * batch_size:(i + 1) * batch_size]
            results = self._model.predict(batch, device=self.device, save=save, conf=self.bounding_box_threshold,
                                          imgsz=self.imgsz,)# half=True)
            results = [result.cpu().numpy() for result in results]
            preds, probs = detection_to_cls_results(results, self.thresholds)
            predictions.extend(preds)
            probabilities.extend(probs)
            if return_boxes:
                boxes.extend(get_boxes(results))
        
        if return_boxes:
            return np.array(predictions), np.array(probabilities), boxes
        return np.array(predictions), np.array(probabilities)
    
if __name__ == '__main__':
    model_path = "YOLO/runs/detect/trainMPS_evenmoredata/weights"
    model = YOLOMultiLabelClassifier(model_path,)
    input_file = "/Users/I538904/gitrepos/Argyrosomus/YOLO/audio/20170116_1130_.wav" # "/Users/I538904/Desktop/convert_to_wav/wav/20170420/2353_.wav"
        #["/Users/I538904/Library/CloudStorage/OneDrive-SAPSE/Portugal/BadData+OtherLoggers/logger-7-MarinaExpo/20230627_200000.WAV"])
        #["/Users/I538904/Library/CloudStorage/OneDrive-SAPSE/Portugal/w_m_lt/1/Montijo_20210712_70140.wav"]
    _, _, boxes = model.predict_file(input_file, save=False, raven_table=True)
    np.savez("runs/boxes.npz", *boxes)
