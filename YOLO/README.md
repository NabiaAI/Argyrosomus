# YOLO Component

This file describes the YOLO method to detect fish vocalizations. It can directly generate bounding boxes along classifications for each box or classify entire segments. An entire segment is multi-label classified as a certain fish, if there is at least one bounding box with confidence exceeding the threshold for this fish. 

The following list outlines the typical workflow to train, evaluate, and use YOLO as object detector or segment classifier:
- The file `create_data_yolo.py` holds several methods to create training and validation datasets as well as a few other utilities. 
  - The most important function is `save_spectrogram`. It turns an audio into a spectrogram and either saves it to disk or returns it as an in-memory buffer. Either way it is in `.png` format. 
  - By using `create_spectrograms`, you can feed in an audio folder which contains audio files together with their raven selection table (`{audio_file_base_path}.wav` and `{audio_file_base_path}.Table.1.selections.txt`). The raven table needs the following columns: `Begin Time (s)`	`End Time (s)`	`Low Freq (Hz)`	`High Freq (Hz)`	`category`, where `category` contains the label (e.g. `lt`, `m`, `w`).  
  The function will create a folder with the spectrograms, with the labels in YOLO format, and a `list.txt` mapping audio to one-hot encoded label. 
  - The function `split_data` then creates a training and validation split of that data that can be used to train YOLO. It also splits the `list.txt` accordingly.
  - Utility functions:
    - `read_audio_file` reads and normalizes an audio from disk.
    - `extract_time_stamp` extracts the time of a recording from the file name and returns it in seconds since 00h00 of that day.
    - `cut_audio_file` and `cut_audio_file_wall_time` cut an audio to a specified time of the recording, or to a specified time of day, respectively. The latter uses the start time of the recording from the file name.
    - `create_long_term_spectrogram` allows reading in a folder of audio files and turn them into a long-term spectrogram by averaging the time-bins of the spectrogram to time bins of a specified length (e.g. averaging over 60 seconds). 
    - `append_audios` turns a folder of individual audios into one large audio.
- `train_yolo.py` then takes the training data and a pre-trained YOLO model (we used `yolo11n.pt` for everything mentioned in the paper; you can get it from the Ultralytics website) and trains the network. It also allows continuing training if anything fails. After training, the thresholds for segment classification are adjusted based on a provided `list.txt`. 
- `evaluate_yolo.py` can then be used to evaluate the model against validation split of the training data and a separate validation set. It plots confusion matrices, roc-curves, provides precision, recall, accuracy, and subset accuracy values as well as an error analysis on the Classify and Count characteristics using that model. 
- `infer_yolo.py` provides the `YOLOMultiLabelClassifier` class which uses YOLO object detection to classify audio segments. It loads a model from a file path. It can then infer individual audio samples or entire files. Additionally, it is possible to have the bounding boxes and total (*not* segment-based) counts for each class (based on bounding boxes) returned. The bounding box object is a list of arrays of bounding boxes with their picture coordinates, class, and confidence for that class. Each list entry (array of bounding boxes) corresponds to one segment.
  - The function `load_cached` loads spectrograms and optionally labels based on several input formats. It can optionally cache them or load them from a cache instead. It sorts out too short audios and pads audios to a target duration if necessary. However, it does not cut them to target duration. The function can take a `ProcessPoolExecutor` to speed up spectrogram generation between calls to `load_cached`. If none is provided, a new one will be allocated using all but one CPU cores. 
  The input can be: 
    - A string file path to a `list.txt` file which maps audio files to labels.
    - A list of numpy arrays and a sampling rate (sr) representing audios. In this case, None will be returned for labels.
  - At the bottom of the file is a usage example that infers a folder of audios and saves the results as raven tables.

`YOLO/yolo_overview.pdf` gives an overview over the previously described steps and general behavior. The light-blue parts are only relevant during training.

# Notes 
- Multi-label outputs are given as arrays of form `(n, 3)`, where `n` is the number of samples classified. Each column stands for a certain fish. The mapping is 0:lt, 1:m, 2:w.
- The labeled data can be found in `labeled_data`.