import os
import io
import numpy as np 
import matplotlib.pyplot as plt
import librosa
import librosa.display
import glob
import scipy.io.wavfile as wav
import pandas as pd
from PIL import Image, ImageDraw
import os
import shutil
from tqdm import tqdm
import random
random.seed(42)
np.random.seed(42)

debug = False
total_labels = 0

segment_duration = 3  # segment duration in seconds
stride = 3  # stride in seconds

# Define the mapping of labels to indices
CATEGORY_MAPPING = {
    "lt": 0,
    "m": 1,
    "w": 2
}

def normalize_audio(audio_data):
    """
    Normalize the given audio data.

    This function converts the audio data to floating point format and normalizes it by dividing 
    by the maximum absolute value in the data. This ensures that the audio data is scaled between 
    -1 and 1. (Required for Librosa.)

    Parameters:
    audio_data (numpy.ndarray): The input audio data to be normalized.

    Returns:
    numpy.ndarray: The normalized audio data in floating point format (required for Librosa).
    """
    # Convert audio data to floating point and normalize
    return audio_data.astype(np.float32) / np.max(np.abs(audio_data))

def append_audios(audio_folder, save=False, normalize=False):
    """
    Append all .wav audio files in a specified folder into a single audio file.
    The order of the audio files is determined by their alphanumeric order.

    Parameters:
    audio_folder (str): The path to the folder containing the .wav files.
    save (bool): If True, save the appended audio data to a new file named 'appended.wav' in the audio folder. Default is False.
    normalize (bool): If True, normalize the appended audio data. Default is False.
    
    Returns:
    tuple: A tuple containing the sample rate (int) and the appended audio data (numpy array).
    
    Raises:
    AssertionError: If there is a sample rate mismatch between the .wav files.
    """
    # Join all wav files in the audio foldeer, save as appended.wav
    # The file order should be alphabetical
    path = os.path.join(audio_folder, "appended.wav")
    if os.path.exists(path):
        os.remove(path)

    # Get a list of all.wav files in the audio folder
    wav_files = glob.glob(os.path.join(audio_folder, "*.wav"))

    # Sort the list of.wav files alphabetically
    wav_files.sort()

    # Initialize an empty list to store the audio data
    audio_data = []

    sample_rate_ref = -1
    # Loop over the sorted.wav files
    for file_path in wav_files:
        # Load the.wav file
        sample_rate, file_data = wav.read(file_path)
        if sample_rate_ref == -1:
            sample_rate_ref = sample_rate
        assert sample_rate == sample_rate_ref, f"Sample rate mismatch: {sample_rate} vs {sample_rate_ref}"

        # Convert to mono if stereo
        if file_data.ndim > 1:
            file_data = np.mean(file_data, axis=1)

        if len(file_data) == 0:
            continue

        # Append the audio data to the list
        audio_data.append(file_data)

    audio = np.concatenate(audio_data)
    if normalize:
        audio = normalize_audio(audio)
    if save:
        # Save the appended audio data to a new.wav file
        wav.write(path, sample_rate, audio)
        print("Appended audio files saved successfully!")
    
    return sample_rate, audio

    
def save_spectrogram(segment, sr, *, file_name="", index=None, as_array = False, save_audio_path = None, averaging_period_s=None, 
                     n_fft=256, hop_length=64, frequency_limit_Hz=1000, image_folder="."):
    """
    Save a spectrogram of an audio segment as a `.png` image file or return it as a numpy array.

    Parameters:
    segment (np.ndarray): The audio segment to process.
    sr (int): The sample rate of the audio segment.
    file_name (str, optional): The base name for the output files. Defaults to "".
    index (int, optional): The index of the segment, used for naming the output files. Defaults to None,
        in which case the file name is not appended with an index.
    as_array (bool, optional): If True, return the spectrogram as a numpy array instead of saving it as a file, 
        otherwise return path to image. Defaults to False.
    save_audio_path (str, optional): The directory to additionally save the audio segment as a wav file. 
        If None, the audio segment is not saved. Defaults to None.
    averaging_period_s (float, optional): The period in seconds over which to average the spectrogram. 
        Useful for long-term spectrograms. Defaults to None.
    n_fft (int, optional): The number of FFT components. Defaults to 256.
    hop_length (int, optional): The number of samples between successive frames. Defaults to 64.
    frequency_limit_Hz (int, optional): The maximum frequency to display in the spectrogram. Defaults to 1000 Hz.
    image_folder (str, optional): The directory to save the spectrogram image. Defaults to ".".

    Returns:
    np.ndarray or str: The spectrogram as a numpy array if as_array is True, otherwise path to image.
    """
    if sr != 4000:
        segment = librosa.resample(segment, orig_sr=sr, target_sr=4000)
        sr = 4000

    # Calculate the spectrogram
    D = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(D, ref=np.max)

    # Define the frequency range to display
    max_freq_bin = min(D.shape[0], int(frequency_limit_Hz / (sr / n_fft)))
    S_db_clipped = S_db[:max_freq_bin, :]
    # flip the image vertically so low frequencies are at the bottom
    S_db_flipped = S_db_clipped[::-1, :]  

    # average the spectrogram over a period of time in seconds along the time dimension (second axis)
    if averaging_period_s:
        averaging_period_samples = int(averaging_period_s * sr / hop_length)
        S_db_flipped = S_db_flipped[:, :-(S_db_flipped.shape[1] % averaging_period_samples)] # trim spectrogram to fit buckets
        S_db_flipped = np.mean(S_db_flipped.reshape(S_db_flipped.shape[0], -1, averaging_period_samples), axis=2)

    # Save the plot as a PNG file
    segment_appendix = f"_segment_{index + 1}" if index is not None else ""
    if as_array:
        output = io.BytesIO()
    else: 
        output = os.path.join(image_folder, f"{file_name}{segment_appendix}.png")
    plt.imsave(output, S_db_flipped, cmap='magma')

    if save_audio_path:
        # Save the audio segment as a wav file
        wav.write(os.path.join(save_audio_path, f"{file_name}{segment_appendix}.wav"), sr, segment)

    if as_array:
        output.seek(0)
        output = np.array(Image.open(output).convert('RGB'))

    return output


def add_bounding_boxes(image_path, segment_start_time, segment_duration, sr, selections, labels_folder, base_name, stride_samples, idx, list_path, segment_audio_path=None):
    """
    Adds bounding boxes to a spectrogram image based on provided Raven selections and saves the updated image and annotations in YOLO format.
    
    Parameters:
    image_path (str): Path to the spectrogram image.
    segment_start_time (float): Start time of the audio segment in seconds.
    segment_duration (float): Duration of the audio segment in seconds.
    sr (int): Sample rate of the audio.
    selections (pd.DataFrame): DataFrame containing ground-truth label information with columns 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', and 'category'.
    labels_folder (str): Directory where the YOLO format label files will be saved.
    base_name (str): Base name for the output files.
    stride_samples (int): Number of samples to stride for each segment.
    idx (int): Index of the current segment.
    list_path (str): Path to the list file where segment information will be appended.
    segment_audio_path (str, optional): Path to the directory containing segment audio files. Defaults to None.
    """
    global total_labels
    # Open the saved spectrogram image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Define the frequency limit for the spectrogram
    freq_limit = 1000  # Hz

    segment_end_time = segment_start_time + segment_duration
    segment_selections = selections[
        (selections['End Time (s)'] > segment_start_time) & 
        (selections['Begin Time (s)'] < segment_end_time)
    ]
    
    # Prepare bounding box data for YOLO format text file
    bounding_boxes = []
    per_image_labels = [0, 0, 0]

    # Draw bounding boxes and labels
    for _, row in segment_selections.iterrows():
        begin_time, end_time = row['Begin Time (s)'], row['End Time (s)']
        low_freq, high_freq = row['Low Freq (Hz)'], row['High Freq (Hz)']
        category = row['category']

        # Get the category index
        category_index = CATEGORY_MAPPING[category]

        # for creating list
        per_image_labels[category_index] = 1

        # Convert time to X-axis coordinates
        image_width, image_height = image.size
        x0 = max((begin_time - segment_start_time) / segment_duration * image_width, 0)
        x1 = min((end_time - segment_start_time) / segment_duration * image_width, image_width)

       # Convert frequency to Y-axis coordinates based on the clipped range
        if high_freq > freq_limit:
            high_freq = freq_limit
        if low_freq > freq_limit:
            continue

        # Calculate Y-axis coordinates based on frequency limits
        y0 = image_height * (1 - low_freq / freq_limit)
        y1 = image_height * (1 - high_freq / freq_limit)

        #print raw box
        # print(f"Raw box: [{x0}, {y0}, {x1}, {y1}] - segment {segment_start_time}")

        # Ensure y0 is always less than y1
        if y0 > y1:
            y0, y1 = y1, y0

        # Check if the bounding box is out of the image
        is_out_of_bounds = (
            x0 < 0 or x1 > image_width or y0 < 0 or y1 > image_height
        )

        if is_out_of_bounds:
            # Only print and process if the box is actually out
            print(f"Box out of bounds before truncation: [{x0}, {y0}, {x1}, {y1}] - segment {segment_start_time}")

            # Truncate bounding box coordinates to fit the image
            x0 = max(x0, 0)
            x1 = min(x1, image_width)
            y0 = max(y0, 0)
            y1 = min(y1, image_height)

            # After truncation, log the updated box
            print(f"Box after truncation: [{x0}, {y0}, {x1}, {y1}] - segment {segment_start_time}")

        # Check again if the box dimensions are invalid after truncation
        if x1 <= x0 or y1 <= y0:
            print(f"Invalid box dimensions after truncation: [{x0}, {y0}, {x1}, {y1}] - segment {segment_start_time}")
            continue


        # Draw rectangle and label
        if debug:
            # Optional debugging statement for valid bounding boxes
            print(f"Valid box: [{x0}, {y0}, {x1}, {y1}] - segment {segment_start_time}")
            draw.rectangle([x0, y0, x1, y1], outline="green", width=2)
            draw.text((x0, y0), category, fill="white")

        # Normalize coordinates for YOLO format (values between 0 and 1)
        x_center = ((x0 + x1) / 2) / image_width
        y_center = ((y0 + y1) / 2) / image_height
        box_width = (x1 - x0) / image_width
        box_height = (y1 - y0) / image_height

        # Append annotation (index, x_center, y_center, box_width, box_height)
        bounding_boxes.append(f"{category_index} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        total_labels += 1
    
    # Save the updated image with bounding boxes
    image.save(image_path)

    with open(list_path, 'a') as f:
        label_str = " ".join([str(x) for x in per_image_labels])
        p = os.path.join(segment_audio_path, f"{base_name}_segment_{(idx)+1}.wav")
        f.write(f"{p}\t{label_str}\n")

    # Image filename and segement index:
    txt_file_path = f"{base_name}_segment_{(idx)+1}.txt"

    # Save the bounding box annotations to a text file
    with open(os.path.join(labels_folder, txt_file_path), "w") as f:
        f.write("\n".join(bounding_boxes))

def copy_files(file_list, dest_images_folder, dest_labels_folder, src_images_folder, src_labels_folder):
    """
    Copies image and label files from source folders to destination folders. Useful for splitting datasets.

    Parameters:
    file_list (list): List of image file names to be copied.
    dest_images_folder (str): Destination folder path for image files.
    dest_labels_folder (str): Destination folder path for label files.
    src_images_folder (str): Source folder path for image files.
    src_labels_folder (str): Source folder path for label files.
    """
    for image_file in file_list:
        # Image file path
        src_image_path = os.path.join(src_images_folder, image_file)
        dest_image_path = os.path.join(dest_images_folder, image_file)

        # Label file path
        label_file = os.path.splitext(image_file)[0] + ".txt"
        src_label_path = os.path.join(src_labels_folder, label_file)
        dest_label_path = os.path.join(dest_labels_folder, label_file)

        # Copy image and label if both exist
        if os.path.exists(src_image_path) and os.path.exists(src_label_path):
            shutil.copy(src_image_path, dest_image_path)
            shutil.copy(src_label_path, dest_label_path)


def split_data(image_folder, labels_folder, list_path):
    """
    Splits the dataset into training and validation sets randomly and copies the files to the respective folders.
    Additionally, it creates a list of audio segments to labels for each set.

    Parameters:
    image_folder (str): Path to the folder containing the images.
    labels_folder (str): Path to the folder containing the labels.
    list_path (str): Path to the file containing the list of audio files and labels.

    Raises:
        AssertionError: If the number of images and labels do not match.
    """
    # Output folders
    output_training_images_folder = "./datasets/to_train/train/images"
    output_training_labels_folder = "./datasets/to_train/train/labels"
    output_validation_images_folder = "./datasets/to_train/valid/images"
    output_validation_labels_folder = "./datasets/to_train/valid/labels"

    # Split ratio
    split_ratio = 0.85

    # Create output folders
    os.makedirs(output_training_images_folder, exist_ok=True)
    os.makedirs(output_training_labels_folder, exist_ok=True)
    os.makedirs(output_validation_images_folder, exist_ok=True)
    os.makedirs(output_validation_labels_folder, exist_ok=True)

    # Get a list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    with open(list_path, 'r') as f:
        list_lines = f.readlines()
        list_lines.sort()

    assert len(list_lines) == len(image_files), f"Number of images and labels do not match: {len(list_lines)} vs {len(image_files)}"

    rnd_idx = np.arange(len(list_lines))
    np.random.shuffle(rnd_idx)
    split_index = int(len(image_files) * split_ratio)


    list_lines = [list_lines[i] for i in rnd_idx]
    image_files = [image_files[i] for i in rnd_idx]

    # save split list
    base_path = os.path.splitext(list_path)[0]
    with open(f"{base_path}_train.txt", 'w') as f:
        f.writelines(list_lines[:split_index])
    with open(f"{base_path}_valid.txt", 'w') as f:
        f.writelines(list_lines[split_index:])

    # Split the dataset
    training_files = image_files[:split_index]
    validation_files = image_files[split_index:]
    # Copy training files
    copy_files(training_files, output_training_images_folder, output_training_labels_folder, image_folder, labels_folder)
    # Copy validation files
    copy_files(validation_files, output_validation_images_folder, output_validation_labels_folder, image_folder, labels_folder)

    print("Data split completed successfully!")

def read_audio_file(file_path):
    """
    Reads an audio file and processes it.
    - If the audio data is stereo, it will be converted to mono by averaging the channels.
    - The audio data will be normalized using the `normalize_audio` function.

    Parameters:
        file_path (str): The path to the audio file.

    Returns:
        tuple: A tuple containing the sample rate (int) and the processed audio data (numpy array).
    """
    sr, audio_data = wav.read(file_path)

    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    audio_data = normalize_audio(audio_data)

    return sr, audio_data

def extract_time_stamp(file_path:str):
    """
    Extracts the timestamp from a given file path and converts it to seconds.

    The function handles different formats of file names to extract the hour and minute information:
    - Files ending with '_.wav' (e.g., '0416_.wav') are interpreted as `hhmm` hours and minutes.
    - Files with a name length of 8 characters where the last 4 characters are digits (e.g., '0415.wav') are interpreted as `hhmm` hours and minutes.
    - Files containing 'log' (e.g., 'log00001.wav') are interpreted as hours with minutes set to '00'. 
        Note files starting with `log` might not contain the correct time information in the filename, 
        as they are just enumerated by the logger. 
    - Files containing '_-' (e.g., '20161117_0757_-08.167-08.500.wav') are interpreted as hours and minutes derived from the decimal part.
    - Other files are assumed to have a format where the hour and minute information is the second part of the filename (e.g., 'date_120000.wav').

    Parameters:
        file_path (str): The path to the file from which to extract the timestamp.

    Returns:
        int: The timestamp in seconds since 00h00 of that day.

    Raises:
        AssertionError: If the extracted hour string is not in the correct format.
    """
    if file_path.endswith('_.wav'):
        idx = file_path.find('_.wav')
        hour_str = file_path[idx-4:idx] # 0416_.wav => 4h 16 min
    elif len(file_path.split('/')[-1]) == 4+4 and file_path[-8:-4].isdigit(): # 0415.wav
        hour_str = file_path[-8:-4]
    elif 'log' in file_path:
        hour_str = file_path.split('.')[-2][-2:] # log00001.wav => 1h; log00022.wav => 22h
        hour_str += '00' # add minutes
    elif '_-' in file_path:
        idx = file_path.find('_-')
        hour_str = file_path[idx+2:idx+4] # 20161117_0757_-08.167-08.500.wav => 08.167 = 8h 10min
        hour_str += str(int(float(file_path[idx+4:idx+7]) / 1000 * 60)).zfill(2) # add minutes
    else:
        hour_str = file_path.split('/')[-1].split('_')[1][:6] # eg date_120000.wav => 120000 = 12h;
    assert hour_str.isdigit() and (len(hour_str) == 6 or len(hour_str) == 4), f"Hour string is not in the correct format: {hour_str}"
    hour = int(hour_str[:2])
    minute = int(hour_str[2:4])
    time_in_seconds = hour * 3600 + minute * 60
    return time_in_seconds

def cut_audio_file_wall_time(file_path, start_wall_clock_time_s, end_wall_clock_time_s, save_base_path=None):
    """
    Cuts a segment from an audio file based on wall clock time (time of the day) and optionally saves it.

    Parameters:
        file_path (str): Path to the input audio file.
        start_wall_clock_time_s (float): Start time in seconds from the wall clock (time of the day).
        end_wall_clock_time_s (float): End time in seconds from the wall clock (time of the day).
        save_base_path (str, optional): Base path to save the cut audio segment. If provided, the segment will be saved
                                        with a filename that includes the start and end times. Defaults to None.

    Returns:
        tuple: A tuple containing the audio segment and the sample rate.
    """
    ts = extract_time_stamp(file_path)

    segment, sr = cut_audio_file(file_path, start_wall_clock_time_s - ts, end_wall_clock_time_s - ts)

    if save_base_path:
        p = f"{save_base_path}-{start_wall_clock_time_s/3600:06.3f}-{end_wall_clock_time_s/3600:06.3f}.wav"
        wav.write(p, sr, segment)

    return segment, sr

def cut_audio_file(file_path, start_time_s, end_time_s, save_base_path=None):
    """
    Cuts a segment from an audio file and optionally saves it to a specified path.

    Parameters:
    file_path (str): Path to the input audio file.
    start_time_s (float): Start time of the segment in seconds.
    end_time_s (float): End time of the segment in seconds.
    save_base_path (str, optional): Base path to save the cut segment. If None, the segment is not saved.

    Returns:
    tuple: A tuple containing the audio segment (numpy array) and the sample rate (int).

    Raises:
    AssertionError: If start_time is negative, end_time exceeds the length of the audio data, or start_time is not less than end_time.
    """
    # Read the audio file
    sr, audio_data = wav.read(file_path)

    # Calculate the start and end time in samples relative to audio
    start_time = int(start_time_s * sr)
    end_time = int(end_time_s * sr)

    assert start_time >= 0 and end_time <= len(audio_data) and start_time < end_time, f"Invalid start or end time: {start_time}, {end_time}, {file_path}"

    # Extract the segment
    segment = audio_data[start_time:end_time]

    if save_base_path:
        p = os.path.join(save_base_path, f"{os.path.basename(file_path)}")
        wav.write(p, sr, segment)

    return segment, sr

def create_spectrograms(audio_folder, image_folder, labels_folder, segment_folder, list_path):
    """
    Generate spectrogram images from audio files and create corresponding labels for YOLO using Raven selection tables as ground truth.
    
    Parameters:
        audio_folder (str): Path to the folder containing audio files (`.wav`) 
            and Raven selection tables (`.Table.1.selections.txt`).
        image_folder (str): Path to the folder where spectrogram images will be saved.
        labels_folder (str): Path to the folder where YOLO label files will be saved.
        segment_folder (str): Path to the folder where audio segments will be saved.
        list_path (str): Path to the file where the list of audio segments and labels will be saved.
    """
    # create working dirs
    os.makedirs(image_folder, exist_ok=True), os.makedirs(labels_folder, exist_ok=True), os.makedirs(segment_folder, exist_ok=True) 

    if os.path.exists(list_path):
        os.remove(list_path)

    for audio_file_name in os.listdir(audio_folder):
        if not audio_file_name.endswith(".wav"):
            continue

        name = os.path.splitext(audio_file_name)[0]
        selections = pd.read_csv(os.path.join(audio_folder, f"{name}.Table.1.selections.txt"), sep='\t')

        images_generated = 0

        file_path = os.path.join(audio_folder, audio_file_name)
        base_name = os.path.splitext(audio_file_name)[0]  # get the base name without extension
        print(f"Processing {audio_file_name}...")

        # Load the .wav file
        sample_rate, audio_data = read_audio_file(file_path)

        # Calculate segment size in samples
        segment_samples = segment_duration * sample_rate
        stride_samples = stride * sample_rate

        # Iterate over segments, save spectrograms, and add bounding boxes
        for i in tqdm(range(0, len(audio_data) - segment_samples + 1, stride_samples)):
            segment = audio_data[i:i + segment_samples]
            segment_start_time = i / sample_rate  # Start time of the current segment
            
            # Save the spectrogram image
            image_path = save_spectrogram(segment, sample_rate, file_name=base_name, index=i // stride_samples, save_audio_path=segment_folder, image_folder=image_folder)
            
            # Add bounding boxes to the saved image
            add_bounding_boxes(image_path, segment_start_time, segment_duration, sample_rate, selections, labels_folder, base_name, stride_samples, i // stride_samples,
                               list_path, segment_audio_path=segment_folder)
            images_generated += 1

def create_long_term_spectrogram(base_path, day):
    """
    Generates and saves a long-term spectrogram for a given day's audio data.
    Uses specific spectrogram parameters suited for long-term analysis.

    Parameters:
    base_path (str): The base directory path where the audio files are located.
    day (str): The specific day (as a string) for which the spectrogram is to be created.
    """
    n_fft, hop_length = 1024, 512 # PARAMETERS ONLY FOR LONG-TERM SPECTROGRAM (hop length = 50% overlap)
    sr, audio = append_audios(os.path.join(base_path, day), normalize=True)
    save_spectrogram(audio, sr, file_name=f"data/{day}_long-term", averaging_period_s=60, n_fft=n_fft, hop_length=hop_length)

if __name__ == '__main__':
    # --- EXAMPLE USAGE ---

    # create long-term spectrogram
    # create_long_term_spectrogram("../convert_to_wav/wav/", "20210427")

    # cut file to specific time and move predictions
    # start_time = 0
    # p = "data/audio/20170418_1953_"
    # cut_audio_file(f"{p}.wav", start_time, 300, "data/audio")
    # df = pd.read_csv(f"{p}.Table.1.selections.txt", sep='\t')
    # df['Begin Time (s)'] -= start_time
    # df['End Time (s)'] -= start_time
    # df.to_csv(f"{p}.Table.1.selections.txt", sep='\t', index=False)

    # cut files to maximum selection:
    # for audio_file_name in os.listdir(audio_folder):
    #     if not audio_file_name.endswith("_.wav"):
    #         continue
    #     name = os.path.splitext(audio_file_name)[0]
    #     selections = pd.read_csv(os.path.join(audio_folder, f"{name}.Table.1.selections.txt"), sep='\t')
    #     latest_selection_end_time = selections['End Time (s)'].max() + 0.5 # add 500 ms
    #     cut_audio_file(os.path.join(audio_folder, audio_file_name), 0, latest_selection_end_time, save_base_path="data/audio2")

    # create data for training (including split train and validation)
    list_path = "./data/train/list.txt"
    audio_folder = "./labeled_data/train/audio"
    image_folder = "./data/train/images"
    labels_folder = "./data/train/labels"
    segment_folder = './data/train/audio_segments'
    create_spectrograms(audio_folder, image_folder, labels_folder, segment_folder, list_path)
    split_data(image_folder, labels_folder, list_path)

    # create data only for validation/testing
    list_path = "./data/validation/list.txt"
    audio_folder = "./labeled_data/validation/audio"
    image_folder = "./data/validation/images"
    labels_folder = "./data/validation/labels"
    segment_folder = './data/validation/audio_segments'
    create_spectrograms(audio_folder, image_folder, labels_folder, segment_folder, list_path)
