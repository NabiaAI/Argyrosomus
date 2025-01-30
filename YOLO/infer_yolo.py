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
from create_data_yolo import save_spectrogram, read_audio_file, normalize_audio, extract_time_stamp

def y_coord_to_freq(coord):
    # coord = freq / (sr / n_fft) # sr / n_fft = is size of each bin
    # => freq = coord * (sr / n_fft)
    sr, n_fft = 4000, 256 # MUST BE EXACTLY THE SAME AS FOR THE SPECTROGRAM CREATION
    return coord * (sr / n_fft)

def x_coord_to_time(coord):
    # coord = coord / (sr / hop_length) # sr / hop_length = number of bins
    sr, hop_length = 4000, 64 # MUST BE EXACTLY THE SAME AS FOR THE SPECTROGRAM CREATION
    return coord / (sr / hop_length) # coord * (sr / hop_length)
    

def convert_to_raven_selection_table(input_array, names, time_shift=0, img_height=64):
    """
    Converts an input array of detection results to a Raven selection table format.

    Parameters:
    input_array (list of lists): A list where each sublist contains detection results in the format 
                                 [x1, y1, x2, y2, category, conf].
    names (list of str): A list of category names corresponding to the category indices.
    time_shift (float, optional): A time shift to apply to the begin and end times. Default is 0.
    img_height (int, optional): The height of the spectrogram image. Default is 64. 
                                Must be exactly the same as during spectrogram creation!

    Returns:
    pd.DataFrame: A DataFrame containing the converted data in Raven selection table format with columns:
                  ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)",
                   "Low Freq (Hz)", "High Freq (Hz)", "category", "confidence"].
    """
    img_height = 64 # RESULTS FROM SPECTROGRAM CREATION 
    data = []
    for idx, row in enumerate(input_array):
        x1, y1, x2, y2, category, conf = row

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
            names[int(category)],
            conf
        ])

    columns = [
        "Selection", "View", "Channel", "Begin Time (s)", "End Time (s)",
        "Low Freq (Hz)", "High Freq (Hz)", "category", "confidence"
    ]
    return pd.DataFrame(data, columns=columns)

def segment_audios(file_paths, segment_duration=5, stride=None, extract_timestamps=True):
    """
    Segments audio files into smaller chunks of specified duration.
    Parameters:
        file_paths (list of str): List of paths to the audio files to be segmented.
        segment_duration (int, optional): Duration of each segment in seconds. Default is 5 seconds.
        stride (int, optional): Step size in seconds between the start of each segment. If None, it defaults to segment_duration. Default is None.
        extract_timestamps (bool, optional): Whether to extract timestamps from the file names. Default is True.
    Returns:
        tuple: A tuple containing:
            - time_stamps (list of float): List of timestamps for each segment.
            - all_segments (list of numpy.ndarray): List of audio segments.
            - sample_rate (int): Sample rate of the audio files. Must be the same for all audios!
    """
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
    """
    Converts object detection predictions to classification results using a threshold.

    More precisely, for each audio segment/input spectrogram it extracts the bounding box 
    with the highest confidence for each (fish) class. If this confidence exceeds the 
    threshold for that class, it is counted as a predicted true classification of that class
    and a predicted false otherwise.

    Parameters:
        preds (list): A list of prediction objects, where each prediction object contains:
            - names (list): A list of class names. Its length is equal to the number of classes.
            - boxes (list): A list of bounding box objects, where each bounding box object contains:
                - cls (array): An array containing the class index.
                - conf (array): An array containing the confidence score.
        thresholds (list, optional): A list of threshold values for each class. If provided, the classification
                                     results will be thresholded based on these values. 
                                     Otherwise they behave as if thresholds were all 0 (any detection is counted).

    Returns:
        tuple: A tuple containing:
            - converted_results (numpy.ndarray): A 2D array of one-hot encoded (thresholded) classification results.
            - converted_probs (numpy.ndarray): A 2D array of classification probabilities for each class 
                                               which is the max confidence of any detected object for that class.
    """
    converted_results = []
    converted_probs = []
    for pred in preds:
        one_hot = np.zeros((len(pred.names),))
        prob = np.zeros((len(pred.names),))
        for b in pred.boxes:
            cls = int(b.cls[0])
            conf = b.conf[0]
            one_hot[cls] = 1
            if conf > prob[cls]: # take the highest confidence (better to optimize threshold)
                prob[cls] = conf
        converted_results.append(one_hot)
        converted_probs.append(prob)
    converted_results = np.array(converted_results)
    converted_probs = np.array(converted_probs)
    if thresholds is not None:
        converted_results = (converted_probs > np.array(thresholds)).astype('int')
    return converted_results, converted_probs

def filter_too_short(audios, labels, srs, min_duration=0.1):
    """
    Filters out audio samples together with their labels that are shorter than a specified minimum duration in seconds.
    
    Parameters:
    audios (list): List of audio samples.
    labels (list or None): List of labels corresponding to the audio samples. If None, no labels are used.
    srs (list): List of sample rates corresponding to the audio samples.
    min_duration (float, optional): Minimum duration (in seconds) for an audio sample to be retained. Default is 0.1 seconds.
    
    Returns:
    tuple: A tuple containing three elements:
        - filtered_audios (list): List of audio samples that meet the minimum duration requirement.
        - filtered_labels (list or None): List of labels corresponding to the filtered audio samples, 
                                          or None if no labels were provided.
        - filtered_srs (list): List of sample rates corresponding to the filtered audio samples.
    """

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
    """
    Converts a numpy array of audio data with sample rate `sr` into a spectrogram.

    Parameters:
    audio (numpy.ndarray): The input audio data as a numpy array.
    sr (int): The sample rate of the audio data.
    target_duration (int, optional): The target duration of the audio in seconds. Default is 5 seconds.

    Returns:
    numpy.ndarray: The spectrogram of the audio data.

    Notes:
    - If the length of the audio is less than the target duration, it will be padded with zeros.
    - If the length of the audio is greater than the target duration, it will be cropped.
    - A warning will be printed if the audio is longer than the target duration.
    """
    if len(audio) < target_duration * sr:
        length = int(target_duration * sr)
        audio = np.pad(audio, (0, length - len(audio)), 'constant')
    if len(audio) > target_duration * sr:
        print(f"WARNING: Audio is longer than target duration -> CROPPING (cut {len(audio) / sr - target_duration:.3}s off)")
        audio = audio[:int(target_duration * sr)]
    spectrogram = save_spectrogram(audio, sr, as_array=True)
    return spectrogram


def load_list(input):
    """
    Load a list of audio files and their corresponding labels from a given input path.

    Parameters:
        input (str): Path to a file or directory. 
                    If a file, it should contain lines with file paths and optional labels separated by a tab. 
                    The labels should be separated by space. A line should therefore look this this:
                    
                    `file_path\t0 1 0`.

                    If a directory, it will recursively search for .wav files. In this case, no labels are returned.

    Returns:  
    tuple: A tuple containing:
        - audio_list (list): List of audio data arrays.
        - labels_list (list): List of labels corresponding to each audio file, or empty list if a directory was provided.
        - srs (list): List of sample rates corresponding to each audio file.

    Notes:
        - If a file path in the input does not exist, it will be skipped and a warning will be printed.
    """
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

def audios_to_spectrograms(audios: list[np.array], srs, target_duration, executor=None, use_tqdm=True): # TODO: split audio if its too long
    """
    Converts a list of audio signals into their corresponding spectrograms using multi-processing.

    Parameters:
        audios (list[np.array]): List of audio signals as numpy arrays.
        srs (list[int]): List of sample rates corresponding to each audio signal.
        target_duration (float): Target duration for each spectrogram.
                Shorter audios will be padded with zeros, longer audios will be cropped.
        executor (concurrent.futures.Executor, optional): ProcessPoolExecutor for parallel processing. 
                Defaults to None (creating one ad-hoc). When repeatedly calling `audios_to_spectrograms`, 
                providing an executor that is only closed after all calls is likely more efficient.
        use_tqdm (bool, optional): Whether to use tqdm for progress tracking. Defaults to True.

    Returns:
        np.array: Array of spectrograms corresponding to the input audio signals.
    """
    spectrogram_list = []
    idxs = []

    close_executor = False
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=os.cpu_count()-1)
        close_executor = True

    futures = {executor.submit(numpy_audio_to_spectrogram, audio, sr, target_duration): idx for idx, (audio, sr) in enumerate(zip(audios, srs, strict=True))}
    iterator = tqdm(futures, total=len(futures)) if use_tqdm else futures
    for future in iterator:
        idx = futures[future]
        spectrogram = future.result()
        spectrogram_list.append(spectrogram)
        idxs.append(idx)

    if close_executor:
        executor.shutdown()

    idx = np.argsort(np.array(idxs)) # sort back to original order
    spectrogram_list = np.array(spectrogram_list)[idx]
    return spectrogram_list

def load_cached(input, cache_path=None, min_duration_s=0.1, target_duration_s=5, no_labels=False, sr=None, executor=None, use_tqdm=True):
    """
    Load cached spectrograms and labels or generate them from audio input.

    Parameters:
    input (str or list): Path to the input file or directory or a list of numpy arrays representing audio data. 
        If input is a file or directory: Refer to `load_list` for more details.
        If input is a list of numpy arrays: A sample rate (parameter `sr`) must be provided.
    cache_path (str, optional): Path to the cache directory. If provided, the function will attempt to load cached data.
    min_duration_s (float, optional): Minimum duration of audio segments to be considered. Default is 0.1 seconds.
    target_duration_s (float, optional): Target duration for each spectrogram. 
        Shorter audios will be padded with zeroes, longer audios will be cropped. Default is 5 seconds.
    no_labels (bool, optional): If True, labels will not be loaded or saved. Default is False.
    sr (int, optional): Sample rate of the audio data. Must be provided if input is a list of numpy arrays.
    executor (concurrent.futures.Executor, optional): Executor for parallel processing. 
        Re-using executor between calls to `load_cached` improves efficiency. Default is None, 
        which creates a new executor for each call.
    use_tqdm (bool, optional): If True, use tqdm for progress visualization. Default is True.

    Returns:
    tuple or numpy.ndarray: If no_labels is False, returns a tuple (spectrogram_list, labels). 
                            If no_labels is True, returns only spectrogram_list.
    """
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

    audio_list, labels, srs = filter_too_short(audio_list, labels, srs, min_duration_s)
    labels = np.array(labels, dtype='int') if labels is not None else None
    spectrogram_list = audios_to_spectrograms(audio_list, srs, target_duration_s, executor=executor, use_tqdm=use_tqdm)

    if cache_path:
        os.makedirs(cache_path, exist_ok=True)
        np.save(f"{cache_path}/spectrograms.npy", spectrogram_list)
        if not no_labels:
            np.save(f"{cache_path}/labels.npy", labels)
    if no_labels:
        return spectrogram_list
    return spectrogram_list, labels

def _get_boxes(results):
    """Small helper function to extract bounding boxes from detection results in a less convoluted format."""
    boxes = []
    for result in results:
        box_list = np.hstack([result.boxes.xyxy, result.boxes.cls.reshape((-1, 1)), result.boxes.conf.reshape((-1, 1))])
        boxes.append(box_list)
    return boxes

class YOLOMultiLabelClassifier:
    """
    A classifier for multi-label classification using YOLO model.
    Attributes:
        bounding_box_threshold (float): The threshold for bounding box confidence.
        device (str): The device to run the model on (one of 'cpu', 'gpu', 'mps').
        iou (float): The Intersection over Union threshold.
        label_indices (dict): A map from label name (e.g. lt) to index (e.g. 0).
        thresholds (list): The thresholds for each class.
    Methods:
        evaluate_and_adjust_thresholds(threshold_train_path, ratio):
            Evaluates and adjusts the thresholds based on the given data.
        predict_file(file_path, *, save=False, raven_table=False, threshold_boxes=False, return_box_predictions=False):
            Predicts the labels for the given audio file.
        predict(input, *, save=False, batch_size=64, return_boxes=False, threshold_boxes=False, return_box_predictions=False):
            Predicts the labels for the given input data (audios as numpy arrays).
    """
    def __init__(self, model_path, *, device='mps', bounding_box_threshold=0.25, thresholds=None):
        """
        Initializes the YOLOMultiLabelClassifier.

        Parameters:
        model_path (str): Path to the model directory or an ultralytics.YOLO model object.
        device (str, optional): The device to run the model on (one of 'cpu', 'gpu', 'mps'). Default is 'mps'.
        bounding_box_threshold (float, optional): The threshold for bounding box confidence. Default is 0.25.
        thresholds (list or str, optional): The thresholds for each class. Default is None.
            - If a string, it is treated as a path to a list file containing audio paths and labels to 
            adjust the threshold based on the F1 optimum (see `evaluate_and_adjust_thresholds`).
            - If a list/array, it is treated as the thresholds directly
            - If None, the thresholds are loaded from the model directory.
        """
        from ultralytics import YOLO
        if isinstance(model_path, str):
            self._model = YOLO(f"{model_path}/best.pt")
        else:
            self._model = model_path
        self.bounding_box_threshold = bounding_box_threshold
        self.device = device
        self.imgsz = (64,320) # based on 4khz audio with 5s duration up to 1000 Hz
        self.iou=0.5
        self.label_indices = {'lt':0, 'm':1, 'w':2}
        self.threshold_meagre_hz = 100

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
        """
        Evaluate and adjust thresholds for each label based on the precision-recall curve and F1 score.
        The threshold maximizing the F1 score is selected.

        The method loads spectrograms and labels from the given list file (see `load_list`).

        Parameters:
            threshold_train_path (str): Path to the training data list file for threshold evaluation (see `load_list`).
            ratio (float): Ratio of the data to be used for evaluation.

        Returns:
            List[float]: A list of optimal thresholds for each label.
        """
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
    
    def predict_file(self, file_path, *, save=False, raven_table=False, threshold_boxes=False, return_box_predictions=False):
        """
        Predicts an audio file of any length by cutting the file into 5 second segments and using `predict` on those segments. 
        Optionally saves the predicted bounding boxes as Raven selection tables next to the input file.

        Parameters:
            file_path (str): Path to the audio file to be processed.
            save (bool, optional): If True, paints the bounding boxes onto the images and saves them.
            raven_table (bool, optional): If True, converts the predictions to a Raven selection table and saves it. Defaults to False.
            threshold_boxes (bool, optional): If True, applies the class thresholds to the predicted boxes. 
                Therefore not saving or returning any bounding boxes that fall below the threshold. Defaults to False.
            return_box_predictions (bool, optional): If True, returns the box predictions (see `predict` for details). Defaults to False.

        Returns:
            tuple: See `predict` for details. The length of the returned arrays is equal to the number of 5-second segments.
        """
        segment_duration = 5
        _, audios, sr = segment_audios([file_path], segment_duration=segment_duration, extract_timestamps=False)
        audios = [normalize_audio(audio) for audio in audios]
        spectrogram = load_cached(audios, cache_path=None, sr=sr, no_labels=True)
        ret = self.predict(spectrogram, save=save, return_boxes=True, threshold_boxes=threshold_boxes, return_box_predictions=return_box_predictions)

        if not raven_table:
            return ret

        time_shift = 0
        boxes = ret[2]
        frames = []
        for b in boxes:
            df = convert_to_raven_selection_table(b, self._model.names, time_shift=time_shift, img_height=spectrogram[0].shape[1])
            if len(df) == 0:
                time_shift += segment_duration
                continue
            df.set_index("Selection", inplace=True)
            frames.append(df)
            time_shift += segment_duration

        df = pd.concat(frames, ignore_index=True) if len(frames) > 0 else df.drop(columns=["Selection"])
        df.reset_index(inplace=True, names="Selection")
        df["Selection"] += 1 # start from 1
        df.to_csv(f"{os.path.splitext(file_path)[0]}.Table.1.selections.predicted.txt", sep="\t", index=False)
        return ret
    
    def filter_meagre_too_low(self, results, img_height):
        if len(results) == 0:
            return results
        
        names = results[0].names
        def should_keep(box):
            if names[int(box.cls[0])] != 'm':
                return True
            _, y1, _, y2 = img_height - box.xyxy[0] # invert y axis so low y = low freq
            min_y = min(y1, y2)
            min_freq = y_coord_to_freq(min_y)
            return min_freq > self.threshold_meagre_hz
 
        for result in results:
            result.boxes = result.boxes[list(map(should_keep, result.boxes))]
        return results

    def predict(self, input, *, save=False, batch_size=64, return_boxes=False, threshold_boxes=False, return_box_predictions=False):
        """
        Predicts the output for the given audio input using the YOLO model. It predicts the bounding boxes 
        and then converts them to classification results using the thresholds (see `detection_to_cls_results`).

        Parameters:
            input (np.ndarray or list): The input data, either as a numpy array or a list of numpy arrays.
            save (bool, optional): Whether to save the predictions. Defaults to False.
            batch_size (int, optional): The size of the batches for prediction. Defaults to 64.
            return_boxes (bool, optional): Whether to return the bounding boxes. Defaults to False.
            threshold_boxes (bool, optional): Whether to threshold the bounding boxes based on confidence. Defaults to False.
            return_box_predictions (bool, optional): Whether to return one-hot encoded predictions for each predicted box. 
                They are not subject to thresholds. Defaults to False.

        Returns:
        tuple: A tuple containing:
        - np.ndarray: The predictions (see `detection_to_cls_results`).
        - np.ndarray: The probabilities (see `detection_to_cls_results`).
        - list (optional): The bounding boxes in (x,y,x,y,cls,conf) format, if return_boxes is True.
        - np.ndarray (optional): The one-hot encoded box predictions. One for each box. Not grouped by segment. 
            Only returned if return_box_predictions is True.
        """
        if isinstance(input, np.ndarray) and input.ndim > 3 or isinstance(input, list) and isinstance(input[0], np.ndarray):
            input = [spectro[...,::-1] for spectro in input]
        
        # transorm into tensors of batch size
        n_batches = math.ceil(len(input) / batch_size)
        predictions = []
        probabilities = []
        boxes = []
        for i in range(n_batches):
            batch = input[i * batch_size:(i + 1) * batch_size]
            results = self._model.predict(batch, device=self.device, save=save, conf=self.bounding_box_threshold, iou=self.iou,
                                          imgsz=self.imgsz,verbose=False, show_labels=False, show_conf=False, line_width=1)# half=True)
            results = [result.cpu().numpy() for result in results]
            results = self.filter_meagre_too_low(results, img_height=batch[0].shape[0])
            preds, probs = detection_to_cls_results(results, self.thresholds)
            predictions.extend(preds)
            probabilities.extend(probs)
            if return_boxes:
                boxes.extend(_get_boxes(results))

        if threshold_boxes:
            for i, box in enumerate(boxes):
                if len(box) == 0:
                    continue
                thresholds = np.array(list(map(lambda x: self.thresholds[int(x)], box[:, -2])))
                boxes[i] = box[box[:, -1].astype('float') > thresholds]

        return_value = (np.array(predictions), np.array(probabilities))
        if return_boxes:
            return_value = return_value + (boxes,)

        if return_box_predictions:
            box_predictions = []
            for box in boxes:
                for b in box:
                    one_hot = np.zeros((len(self.label_indices),))
                    one_hot[int(b[-2])] = 1
                    box_predictions.append(one_hot)
            box_predictions = np.array(box_predictions) if len(box_predictions) > 0 else np.zeros((1, len(self.label_indices)))
            return_value = return_value + (box_predictions,)

        return return_value
    
if __name__ == '__main__':
    model_path = "YOLO/final_model/weights"
    model = YOLOMultiLabelClassifier(model_path,)

    # base = "YOLO/data/validation/"
    # for f in sorted(os.listdir(base)):
    #     if f.endswith(".wav"):
    #         input_file = os.path.join(base, f)
    #         _, _, boxes = model.predict_file(input_file, save=False, raven_table=True, threshold_boxes=True)
    input_file = "convert_to_wav/wav/20160508/0301_.wav"
    _, _, boxes = model.predict_file(input_file, save=False, raven_table=True, threshold_boxes=True)
