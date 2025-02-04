import sys
import os
import shutil
import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)
sys.path.append("YOLO"), sys.path.append("CNN")
import argparse
import functools
from CNN.macls.trainer_mlabel import MAClsTrainer
import quantification as quant
from CNN.macls.utils.utils import add_arguments
from CNN.macls.data_utils.audio import AudioSegment
from scipy.stats import linregress
from scipy.interpolate import make_interp_spline
import utils_eval as eval
import pymannkendall 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import pandas as pd
from YOLO.create_data_yolo import normalize_audio
from YOLO.infer_yolo import YOLOMultiLabelClassifier, load_cached, segment_audios
from concurrent.futures import ProcessPoolExecutor
    
def _infer_cnn(audios, sample_rate):
    """
    Perform inference using the CNN model on a list of audio segments.

    Args:
        audios (list): A list of audio data arrays.
        sample_rate (int): The sample rate of the audio data.

    Returns:
    tuple: A tuple containing:
        - preds (list): The predicted labels or outputs from the CNN model.
        - outs (list): The raw outputs from the CNN model.
        - None: Placeholder for additional return values (currently None).
    """
    audios = [AudioSegment.from_ndarray(audio, sample_rate) for audio in audios]
    outs, preds = cnn_model.predict(audios,resume_model=args.resume_model)
    return preds,outs,None

def _infer_yolo(audios, sample_rate):
    """
    Perform YOLO inference on a list of audio samples.

    This function normalizes the input audio samples, creates spectrograms using a 
    persistent ProcessPoolExecutor for efficiency, and performs batch inference using a YOLO model.

    Parameters:
        audios (list): List of audio samples to be processed.
        sample_rate (int): The sample rate of the audio samples.

    Returns:
    tuple: A tuple containing:
        - np.array: Array of predictions for each audio sample.
        - list: List of bounding boxes for each prediction.
        - list: List of total vocaliztions for each audio sample.
    """
    audios = [normalize_audio(audio) for audio in audios]
    # persistant executor to speed up spectrogram creation
    executor = ProcessPoolExecutor(max_workers=os.cpu_count()-1)
    # bach inference
    batch_size = 1024 + 512
    n_batches = math.ceil(len(audios) / batch_size)
    preds = []
    boxes = []
    box_preds = []
    for i in tqdm(range(n_batches)):
        batch = audios[i * batch_size:(i + 1) * batch_size]
        batch = load_cached(batch, cache_path=None, sr=sample_rate, no_labels=True, executor=executor, use_tqdm=False)
        batch_preds, _, batch_boxes, box_preds_batch = yolo_model.predict(batch, save=False, batch_size=64, return_boxes=True, return_box_predictions=True)
        preds.extend(batch_preds)
        boxes.extend(batch_boxes)
        box_preds.extend(box_preds_batch)
    return np.array(preds), boxes, box_preds

def _save_array_list(path, arrays: list[np.ndarray]):
    """
    Save a list of numpy arrays to a compressed .npz file as a single array, with each array 
    prefixed by its index in the list. Empty arrays are ignored and not saved explicitly.
    Therefore, if an index is missing, the corresponding array is empty (arr.size == 0).

    Parameters:
    path (str): The file path where the .npz file will be saved.
    arrays (list of np.ndarray): A list of numpy arrays to be saved. Each array 
                                 will be prefixed with its index in the list 
    """
    arrays_with_idx = []
    for i, array in enumerate(arrays):
        if array.size == 0:
            continue
        arrays_with_idx.append(np.hstack([np.ones((array.shape[0], 1)) * i, array]))
    array_with_idx = np.vstack(arrays_with_idx, dtype=np.float32) if len(arrays_with_idx) > 0 else np.array([], dtype=np.float32)
    np.savez_compressed(path, boxes_with_segment_idx=array_with_idx)

def infer(path: str, out_path=None, skip_existing=True, save_box_preds=False, extract_timestamps=True):
    """
    Perform inference on audio files and save the results.

    Parameters:
    path (str): Path to the input file or directory containing audio files. 
                It can be one of the following: 

                - A .txt file containing a list of paths to audio files. The output name will be the name of the .txt file. 
                - A .wav file. The output name will be the name of the .wav file. 
                - A directory containing .wav files. The output name will be the name of the directory.
    out_path (str): Path to the directory where output files will be saved.
    skip_existing (bool, optional): If True, skip processing if output files already exist. Default is True.
    save_box_preds (bool, optional): If True, save box predictions, which are total vocalizations. Default is False.
    extract_timestamps (bool, optional): If True, extract timestamps from the audio files. 
                If False, 0s since begin of the day is assumed. Default is True.
    Returns: (preds, times) or (None, None) if inference is skipped.
    """
    if os.path.isfile(path):
        if path.endswith('.txt'):
            with open(path) as f:
                lines = f.readlines()
            paths = [line.strip().split('\t')[0] for line in lines]
            name = os.path.basename(path)
        if path.endswith('.wav'):
            paths = [path]
            name = os.path.splitext(os.path.basename(path))[0]
    else: 
        paths = _get_full_path_of_audio_files(path).values()
        name = path.split('/')[-1]
        if name.startswith('.'):
            print(f"Skipping {name} as it is a hidden directory")
            return None, None
    paths = sorted(paths)
    
    print("processing", name)
    if out_path is None:
        times, audios, sample_rate = segment_audios(paths, extract_timestamps=extract_timestamps)
        preds, _, _ = _infer_model(audios, sample_rate)
        return preds, times

    out_times, out_preds, out_boxes, out_bpreds = f'{out_path}/{name}_times.npz', f'{out_path}/{name}_preds.npz', f'{out_path}/{name}_boxes.npz', f'{out_path}/{name}_boxpreds.npy'
    if os.path.exists(out_times) and os.path.exists(out_preds) and skip_existing:
        print(f"Skipping {name} as output files already exist")
        return None, None

    times, audios, sample_rate = segment_audios(paths)

    preds, boxes, box_preds = _infer_model(audios, sample_rate)
    _save_array_list(out_boxes, boxes)
    np.savez_compressed(out_times, times=times)
    np.savez_compressed(out_preds, preds=preds)
    if save_box_preds:
        np.save(out_bpreds, box_preds)
    return preds, times

def _get_full_path_of_audio_files(audio_path):
    audio_files = {}
    for root, _, files in os.walk(audio_path):
        for file in files:
            if not file.lower().endswith('.wav'):
                continue
            assert file not in audio_files, f"Duplicate file {file}"
            audio_files[file] = os.path.join(root, file)
    return audio_files

def _average_nan_partially_ignored(arr: np.ndarray, window_size):
    """
    Smooths an array by averaging values within a specified window size, partially ignoring NaN values.

    Parameters:
    arr (np.ndarray): Input array containing numerical values, possibly with NaNs.
    window_size (int): Size of the window to use for averaging.

    Returns:
    np.ndarray: Smoothed array with the same shape as the input array, where each element is the average 
                of the valid (non-NaN) values within the window centered on that element. If the number 
                of valid values within the window is less than or equal to half the window size, the 
                corresponding element in the output array will be NaN.
    """
    arr_smooth = np.ones_like(arr) * np.nan
    half_window = window_size // 2
    for i in range(len(arr)):
        # Define the window range
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window_values = arr[start:end]
        valid_values = window_values[~np.isnan(window_values)] # Exclude np.nan values
        if len(valid_values) > half_window:
            arr_smooth[i] = np.mean(valid_values)
    return arr_smooth

def _smooth_and_plot(x, y, window_size, label, color, only_moving_average=False):
    """
    Smooths and plots the input data using a moving average and optionally plots the smoothed data using spline interpolation.
    
    Parameters:
    x (array-like): The x-coordinates of the data points.
    y (array-like): The y-coordinates of the data points.
    window_size (int): The size of the window for computing the moving average.
    label (str): The label for the plot.
    color (str): The color of the plot line.
    only_moveing_average (bool, optional): If True, only the moving average is plotted.
                                           Otherwise a B-Spline interpolation is added. Defaults to False.
    """
    # Compute the moving average
    y_smooth = _average_nan_partially_ignored(y, window_size)
    x_smooth = _average_nan_partially_ignored(x, window_size)

    if only_moving_average:
        plt.plot(x_smooth, y_smooth, color=color, label=label)
        return
    
    # Downsample the data to reduce the number of knots
    num_knots = len(x_smooth) // 2
    indices = np.linspace(0, len(x_smooth) - 1, num_knots).astype(int)
    x_smooth = x_smooth[indices]
    y_smooth = y_smooth[indices]

    # Create smoothed lines using spline interpolation
    xnew = np.linspace(x_smooth.min(), x_smooth.max(), 300)
    spl = make_interp_spline(x_smooth, y_smooth, k=3)  # k=3 for cubic spline
    y_smooth_spline = spl(xnew)

    # Plot the smoothed line
    plt.plot(xnew, y_smooth_spline, color=color, label=label, alpha=0.3)

def _plot_analysis(all_ratios, all_m_w_counts, n_boot, batch_number, path, file_ending='.pdf'):
    """
    Plots analysis of counts with optional bootstrapping.

    Parameters:
    all_ratios (numpy.ndarray): Array of ratios to be plotted.
    all_m_w_counts (numpy.ndarray): Array of counts corresponding to the ratios.
    n_boot (int): Number of bootstrap samples. If 1, no bootstrapping was performed for all_ratios and all_m_w_coutns.
    batch_number (int): Batch number used to determine the number of samples.
    path (str): Path to save the plot.
    file_ending (str, optional): File extension for the saved plot. Default is '.pdf' (vector graphic).
    """
    if n_boot == 1:
        chosen_idxes = np.arange(all_ratios.shape[0])
    else: 
        chosen_idxes = np.random.choice(all_ratios.shape[0], min(20, n_boot) * batch_number)
    plt_ratios = all_ratios[chosen_idxes]
    _, ax1 = plt.subplots(figsize=(10,8))
    #ax1.scatter(plt_ratios[:,0], plt_ratios[:,1], label='Ratios')
    #ax1.plot(plt_ratios[:,0], intercept + slope * plt_ratios[:,0], label='Trend Line', linestyle='--')
    ax1.set_xlabel('Time (h)'), #ax1.set_ylabel('Ratio (m/w)'), ax1.legend(loc='upper left')
    ax1.set_xticks(np.arange(0, 24, 2))

    # scatter-plot counts
    #ax2 = ax1.twinx()
    ax2 = ax1 # REMOVE THIS TO PLOT BOTH
    plt_lt_m_w = all_m_w_counts[chosen_idxes]
    ax2.scatter(plt_ratios[:,0], plt_lt_m_w[:,0], label='lt', marker="o", color='orange')
    ax2.scatter(plt_ratios[:,0], plt_lt_m_w[:,1], label='m', marker="x", color='green')
    ax2.scatter(plt_ratios[:,0], plt_lt_m_w[:,2], label='w', marker="+", color='red')

    # Plot smoothed lines 
    window_size = 10  # Adjust the window size as needed
    _smooth_and_plot(plt_ratios[:,0], plt_lt_m_w[:,0], window_size, None, 'orange', only_moving_average=True)
    _smooth_and_plot(plt_ratios[:,0], plt_lt_m_w[:,1], window_size, None, 'green', only_moving_average=True)
    _smooth_and_plot(plt_ratios[:,0], plt_lt_m_w[:,2], window_size, None, 'red', only_moving_average=True)

    ax2.set_ylabel('Counts'), ax2.legend(fontsize='small')
    plt.grid(axis='x', alpha=0.2)

    #plt.xticks(np.arange(batch_number), np.arange(batch_number) + 2015)
    #CI_str = f"(95%CI: [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}])"if n_boot > 1 else ""
    #plt.text(0.03, 0.83, f'r: {r_value:.3f} (r²: {r_value**2:.3f}), slope {slope:.3f} {CI_str}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    #plt.text(0.03, 0.79, trend_str, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.savefig(f'{path}_daily{file_ending}', bbox_inches='tight', pad_inches=0)
    plt.close()

def aggregate_over_time(preds:np.ndarray, times:np.ndarray, n_boot=1, aggregation_interval_minutes = 10, fill_missing_with_NaN=False, min_time=None, max_time=None, verbose=False):
    """
    Sums `n` predictions over specified time intervals.

    Parameters:
    preds (np.ndarray): Array of predictions with shape=(n, 3).
    times (np.ndarray): Array of corresponding times shape=(n,).
    n_boot (int, optional): Number of bootstrap samples. Default is 1.
    aggregation_interval_minutes (int, optional): Interval for aggregation in minutes. Default is 10.
    fill_missing_with_NaN (bool, optional): Whether to fill missing time intervals with NaN. Default is False.
    min_time (int, optional): Minimum time for aggregation. Default is None, which uses the minimum time in `times`.
    max_time (int, optional): Maximum time for aggregation. Default is None, which uses the maximum time in `times`.
    verbose (bool, optional): Whether to display progress. Default is False.

    Returns:
    tuple: A tuple containing:
        - all_ratios (list): List of aggregated ratios with shape=(n, n_boot,).
        - all_lt_m_w_counts (list): List of aggregated counts with shape=(n, n_boot, 3).
        - all_times (list): List of mean times for each aggregation interval with shape=(n,).
    """
    assert len(preds) == len(times)
    assert (not np.isnan(times).any()), "Times contain NaNs"

    idx = np.argsort(times)
    preds = preds[idx]
    times = times[idx]

    aggr_interval_s = aggregation_interval_minutes * 60
    all_ratios = []
    all_lt_m_w_counts = []
    all_times = []
    if min_time is None: min_time = int(times.min())
    if max_time is None: max_time = int(times.max())

    iterator = enumerate(range(min_time, max_time, aggr_interval_s))
    if verbose:
        iterator = tqdm(list(iterator))

    max_segments = aggr_interval_s//seg_dur_s

    lb = 0 # lower bound search space; this is done to improve efficiency by not searching the entire array
    for i, t in iterator:
        ub = lb + max_segments * 30 # upper bound search space (give it some leeway)
        mean_time = np.array([t, t + aggr_interval_s]).mean()
        batch_times = (t <= times[lb:ub]) & (times[lb:ub] < t + aggr_interval_s)
        indices = np.where(batch_times)[0]

        if len(indices) == 0: #not np.any(batch_times):
            if fill_missing_with_NaN:
                all_times.append(mean_time)
                all_ratios.append(np.ones((n_boot,)) * np.nan)
                all_lt_m_w_counts.append(np.ones((n_boot, 3)) * np.nan)
            continue

        # Do not change the order of the next 4 lines light-heartedly
        max_idx = indices.max()
        indices = indices[:max_segments] # limit to 1 sample per seg_dur_s seconds
        batch = preds[lb:ub][indices]
        lb += max_idx + 1

        if n_boot == 1:
            lt_m_w_counts = batch.sum(axis=0, keepdims=True)
            ratios = lt_m_w_counts[:, 1] / lt_m_w_counts[:, 2]
        else:
            lt_m_w_counts, ratios = quant.bootstrap(preds=batch, which_ratio=[1,2], n=n_boot)
        all_times.append(mean_time)
        all_ratios.append(ratios)
        all_lt_m_w_counts.append(lt_m_w_counts)
    return all_ratios, all_lt_m_w_counts, all_times

def _analyze_trend_ratios(all_ratios, all_times, n_boot=1):
    """
    Analyzes statistical trends in the provided ratio data using linear regression and Mann-Kendall test.
    Parameters:
    all_ratios (list of arrays): A list where each element is a ratio of m/w corresponding to different times.
    all_times (list): A list of time points corresponding to the data in all_ratios.
    n_boot (int, optional): Number of bootstrap samples used to generate confidence interval estimation. Default is 1.
    Returns:
    None: This function prints the results of the statistical analysis, including the Mann-Kendall test results,
          linear regression slope, confidence intervals, and trend interpretation.
    """
    all_ratios = [np.vstack([np.ones_like(ratios) * time / 3600, ratios]).T for time, ratios in zip(all_times, all_ratios)]

    slopes = []
    ratios_3d = np.array(all_ratios)
    for i in range(0, n_boot):
        slope, _, _, _, _ = linregress(ratios_3d[:,i,0], ratios_3d[:,i,1])
        slopes.append(slope)
    
    mannkendall = pymannkendall.original_test([r[:,1].mean() for r in all_ratios])
    print(f"Mann-Kendall monotony test: trend: {mannkendall.trend}, tau={mannkendall.Tau:.3f}, p-value={mannkendall.p:.3f}, slope={mannkendall.slope:.3f}")

    all_ratios = np.vstack(all_ratios)
    slopes = np.array(slopes)
    slope_ci = (np.percentile(slopes, 2.5), np.percentile(slopes, 97.5))

    slope, intercept, r_value, p_value, _ = linregress(all_ratios[:,0], all_ratios[:,1])

    if n_boot > 1:
        print(f"95% CI slope: [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]")
        print(f"Mean slope: {np.mean(slopes):.3f}, std: {np.std(slopes):.3f}, r-value: {r_value:.3f}")
        trend_str = "Trend: " + ("increasing" if slope_ci[0] > 0 else "decreasing" if slope_ci[1] < 0 else "no significant trend")
    else:
        print(f"Slope: {slope:.3f}, r-value: {r_value:.3f}, p-value: {p_value:.3f}")
        trend_str = "Trend: " + ("increasing" if slope > 0 and p_value < 0.05 else "decreasing" if slope < 0 and p_value < 0.05 else "no significant trend") + f" (p: {p_value:.3f})"  
    print(trend_str)

def analyze_by_day(in_path: str, n_boot=1, aggregation_interval_minutes=10, start_time_s=0, file_ending='.pdf'):
    """
    Analyzes prediction data by day and performs plotting of day. If the input path is a `.wav` file,
    the function infers prediction and time data directly. Otherwise, it loads the prediction and time data.
    If input is `.wav` file, the start time of the audio can be specified in seconds.

    The function performs the following steps:
    1. Loads prediction and time data from .npz files or infers them directly if in_path is `.wav` file.
    2. Filters out invalid times that are not within a day.
    3. Aggregates data over time and calculates ratios.
    4. Plots the analysis results as graph of counts over the day.

    Parameters:
    in_path (str): The input path to the prediction and time data files.
    n_boot (int, optional): The number of bootstrap samples to use for slope estimation. Default is 1.
    aggregation_interval_minutes (int, optional): The time interval in minutes for aggregating the data. Default is 10.
    start_time_s (int, optional): The start time (of the audio) in seconds of the day. Only active if in_path is `.wav`,
        since otherwise times were extracted from the file name during inference. Default is 0.
    file_ending (str, optional): The file extension for the saved plot. Default is '.pdf' (vector graphic).

    Returns:
    np.ndarray: An array of counts of each fish species over time intervals. Shape is (n, 3) where n is 
    the number of time intervals.
    """
    print(f"Analyzing {in_path}")
    if in_path.lower().endswith('.wav'):
        preds, times = infer(path=in_path, out_path=None, skip_existing=True, extract_timestamps=False)
        times = np.array(times) + start_time_s
        in_path = in_path[:-len('.wav')]
    else:
        preds = np.load(f'{in_path}_preds.npz')['preds'] # shape (n, 3)
        times = np.load(f'{in_path}_times.npz')['times'] # shape (n,) in seconds of the day
    if len(preds) == 0:
        print(f"No preds in {in_path}")
        return np.zeros((1, 3))
    valid_times = (0 <= times) & (times < 23 * 3600 + 59 * 60 + 59) # only keep times within a day
    preds = preds[valid_times]
    times = times[valid_times]

    all_ratios, all_lt_m_w_counts, all_times = aggregate_over_time(preds, times, n_boot=n_boot, 
                                                                   aggregation_interval_minutes=aggregation_interval_minutes)
    batch_number = len(all_times)

    # _analyze_trend_ratios(all_ratios, all_times, n_boot=n_boot)

    # append times to ratios for plotting
    all_ratios = [np.vstack([np.ones_like(ratios) * time / 3600, ratios]).T for time, ratios in zip(all_times, all_ratios)]
    all_ratios = np.vstack(all_ratios)
    all_lt_m_w_counts = np.vstack(all_lt_m_w_counts)

    _plot_analysis(all_ratios, all_lt_m_w_counts, n_boot, batch_number, path=in_path, file_ending=file_ending)
    return all_lt_m_w_counts

def infer_all(in_path, out_path, skip_existing=True):
    """
    Perform inference on all directories within the given input path.

    Parameters:
        in_path (str): The input directory path containing subdirectories to process.
        out_path (str): The output directory path where results will be saved.
        skip_existing (bool, optional): If True, skip processing for directories that already have results. Defaults to True.
    """
    for root, dirs, _ in os.walk(in_path):
        dirs = sorted(dirs)
        for d in tqdm(dirs):
            p = os.path.join(root, d)
            infer(path=p, out_path=out_path, skip_existing=skip_existing)

def _get_true_counts(path_to_selection_table):
    """
    Calculate the true counts of fish sounds in segments of seg_dur_s seconds from a Raven selection table.

    Parameters:
        path_to_selection_table (str): The file path to the selection table in CSV format.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - The first array contains the binary presence (0 or 1) of each category ('lt', 'm', 'w') 
              across all segments.
            - The second array contains the total counts of each category ('lt', 'm', 'w') across all segments.
    """
    label_idx_mapping = {
        'lt': 0,
        'm': 1,
        'w': 2
    }
    df = pd.read_csv(path_to_selection_table, sep='\t')
    if len(df) == 0:
        return np.array([0,0,0]), np.array([0,0,0])

    labels = []
    for start_time in range(0, int(df['End Time (s)'].max())+10, seg_dur_s): # loop over segments of seg_dur_s seconds
        end_time = start_time + seg_dur_s
        segment = df[
            (df['End Time (s)'] > start_time) & 
            (df['Begin Time (s)'] < end_time)
        ]
        categories = segment['category']
        label = np.array([0, 0, 0])
        for c in categories:
            label[label_idx_mapping[c]] += 1
        labels.append(label)
    labels = np.array(labels)
    return np.clip(labels, 0, 1).sum(axis=0), labels.sum(axis=0)

def analyze_all_days_by_day(in_path):
    """
    Analyzes data for all `n` days in the given input directory. Each day is analyzed by day.

    Parameters:
        in_path (str): The input directory path containing the day directories (format YYYYMMDD) to be analyzed.

    Returns:
        tuple: A tuple containing:
            - dates (numpy.ndarray): An array of dates corresponding to the processed day result files. Shape =(n,).
            - sums (numpy.ndarray): An array of average counts over time slots for each day. Shape=(n, 3).
    """
    dates=[]
    sums = []
    runs = 0
    for root, _, files in os.walk(in_path):
        assert runs == 0, "Only one level is supported"
        files = sorted(files)
        for f in files:
            if f.endswith('_preds.npz'):
                f: str = f.replace('_preds.npz', '')
                p = os.path.join(root, f)
                counts = analyze_by_day(p)
                dates.append(f)
                sums.append(counts.sum(axis=0)/len(counts)) # average over time slots (negates the effect if a day has less data)
    sums = np.array(sums)
    dates = np.array(dates)
    return dates, sums

def plot_against_validation_data(validation_path, dest = f"data/validation_analyzed", skip_existing=False):
    """
    Plots analyzed audio files against ground-truth validation data 
    and generates comparison plots based on segmented and total count.

    Parameters:
    validation_path (str): The path to the directory containing audio files along with ground-truth Raven selection tabels.
    dest (str, optional): The destination directory to save plots. Defaults to "data/validation_analyzed".
    skip_existing (bool, optional): If True, skips processing files that already exist in the destination directory. Defaults to False.
    Raises:
    AssertionError: If the corresponding Raven selection table file for an audio file is not found.
    """
    for f in sorted(os.listdir(validation_path)):
        p = os.path.join(validation_path, f)
        if f.endswith('.wav'):
            infer(path=p, out_path=dest, skip_existing=skip_existing, save_box_preds=True)
        if f.endswith('Table.1.selections.txt'):
            shutil.copy(p, os.path.join(dest, f))

    dates = []
    sums = []
    true_counts = []
    true_counts_total = []
    box_preds_sums = []
    for f in sorted(os.listdir(dest)):
        p = os.path.join(dest, f)
        if f.endswith('_preds.npz'):
            preds = np.load(p)['preds'] # shape (n, 3)
            box_preds = np.load(f"{p[:p.index('_preds.npz')]}_boxpreds.npy")
            sums.append(preds.sum(axis=0))
            box_preds_sums.append(box_preds.sum(axis=0))

            raven_file = f'{p[:p.index('_preds.npz')]}.Table.1.selections.txt'
            assert os.path.exists(raven_file), f"Labels not found for {p}"
            segment_counts, total_counts = _get_true_counts(raven_file)
            true_counts.append(segment_counts), true_counts_total.append(total_counts)
            dates.append(f)
    
    sums, true_counts = np.array(sums), np.array(true_counts)
    box_preds_sums, true_counts_total = np.array(box_preds_sums), np.array(true_counts_total)
    dates = list(map(lambda x: f"{x[15:17]}h-{x[6:8]}.{x[4:6]}.{x[2:4]}", dates))

    eval.plot_validation_output(dates, sums, true_counts, save_path=f"{dest}/result_segment.pdf")
    eval.plot_validation_output(dates, box_preds_sums, true_counts_total, save_path=f"{dest}/result_total_calls.pdf")


def plot_over_time(in_path, use_cached=False, aggr_interval_min=10, only_every_nth_day=1, smoothing_window=1):
    """
    The function reads result files (format 'YYYYMMDD') from the specified input path, processes it to 
    calculate sums for each day, formats the dates, and then plots the sums 
    against the dates. The resulting plot is saved as 'over_years.pdf' in the 
    input path directory.

    If `use_cached` is True, the function uses cached results from `compute_count_over_years` 
    and uses `in_path` as the path to find the results.

    Parameters:
    in_path (str): The input path where result_files (format 'YYYYMMDD') reside. 
                    Dates are extracted from the file names.
    use_cached (bool, optional): If True, uses cached results from `compute_count_over_years`. Default is False.
    aggr_interval (int, optional): The time interval in minutes for aggregating the data. 
                                    Must match results form `compute_counter_over_years`. Default is 10 minutes. 
    only_every_nth_day (int, optional): The interval for plotting dates. Default is 1, i.e. every day.
    smoothing_window (int, optional): The window size for smoothing the data line plot. Default is 1, i.e. no smoothing.
    """
    if use_cached:
        lt_m_w_counts = np.load(f'{in_path}/all_lt_m_w_counts.npy')
        sums = [_transform_time_series_to_dial_plot(lt_m_w_counts[:,i], 24*60//aggr_interval_min).sum(axis=0) for i in range(lt_m_w_counts.shape[1])] 
        sums = np.array(sums).T
        times = np.load(f'{in_path}/all_times.npy')
        times = _transform_time_series_to_dial_plot(times, 24*60//aggr_interval_min) 
        dates = list(datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d') for d in times.mean(axis=0))
    else:
        sums, dates = analyze_all_days_by_day(in_path)
        dates = list(map(lambda x: x[:4] + '.' + x[4:6] + '.' + x[6:8], dates))

    # plot the sums against the dates
    plt.figure(figsize=(16*2, 4*2))
    datex_idx = list(range(0,len(dates),only_every_nth_day))
    plt.scatter(datex_idx, sums[:,0][::only_every_nth_day], label='lt', marker="o", color='orange')
    plt.scatter(datex_idx, sums[:,1][::only_every_nth_day], label='m', marker="x", color='green')
    plt.scatter(datex_idx, sums[:,2][::only_every_nth_day], label='w', marker="+", color='red')
    _smooth_and_plot(np.arange(len(dates)), sums[:,0], smoothing_window, None, 'orange', only_moving_average=True)
    _smooth_and_plot(np.arange(len(dates)), sums[:,1], smoothing_window, None, 'green', only_moving_average=True)
    _smooth_and_plot(np.arange(len(dates)), sums[:,2], smoothing_window, None, 'red', only_moving_average=True)
    plt.xticks(list(range(0,len(dates),365))+[len(dates)-1], dates[::365]+[dates[-1]], rotation=90)
    plt.xticks(np.arange(0, len(dates), 365/12), minor=True)
   # plt.yscale('log')
    plt.legend()
    plt.savefig(f'{in_path}/over_years.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

def _get_epoch_time(date_str):
    """Returns a unix timestamp from a date string in the format 'YYYYMMDD'."""
    date = datetime.datetime.strptime(date_str, "%Y%m%d")
    return int(date.timestamp())

def compute_count_over_years(in_path, result_path, aggregation_interval_minutes=10):
    """
    Computes and aggregates prediction counts (segmented count) over time for result files found in in_path.
    Missing data is filled with NaN.

    Parameters:
    in_path (str): The input directory path containing the prediction and time files.
    result_path (str): The output directory path where the aggregated results will be saved.
    aggregation_interval_minutes (int, optional): The time interval in minutes for aggregating the data. Default is 10 minutes.

    Notes:
    - The input files should be in the format 'YYYYMMDD_preds.npz' and 'YYYYMMDD_times.npz'.
    - The function assumes that the prediction files contain an array named 'preds' with shape (n, 3).
    - The function assumes that the time files contain an array named 'times' with shape (n,) representing seconds of the day.

    Saves:
    - 'all_lt_m_w_counts.npy': Aggregated prediction counts over the specified time intervals.
    - 'all_times.npy': Corresponding times for the aggregated prediction counts.
    """
    all_days_preds = []
    all_days_times = []
    min_day = 10_000_000_000
    max_day = 0
    runs = 0
    for root, _, files in os.walk(in_path):
        assert runs == 0, "Only one level is supported"
        files = sorted(files)
        files = list(filter(lambda x: x.endswith('_preds.npz'), files))
        for f in tqdm(files, total=len(files)):
            f: str = f.replace('_preds.npz', '')
            day_time_unix = _get_epoch_time(f) # f is in YYYYMMDD format
            p = os.path.join(root, f)
            preds = np.load(f'{p}_preds.npz')['preds'] # shape (n, 3)
            times = np.load(f'{p}_times.npz')['times'] # shape (n,) in seconds of the day
            if preds.size == 0:
                print(f"No data in {p}")
                continue
            times += day_time_unix
            min_day = min(min_day, day_time_unix)
            max_day = max(max_day, day_time_unix)
            all_days_preds.append(preds)
            all_days_times.append(times)

    all_days_preds = np.vstack(all_days_preds)
    all_days_times = np.hstack(all_days_times)
    max_time = max_day+24*3600 
    _, all_lt_m_w_counts, all_times = aggregate_over_time(all_days_preds, all_days_times, fill_missing_with_NaN=True, 
                                                          min_time=min_day, max_time=max_time , aggregation_interval_minutes=aggregation_interval_minutes,
                                                          verbose=True)
    all_lt_m_w_counts, all_times = np.vstack(all_lt_m_w_counts), np.array(all_times)

    np.save(f'{result_path}/all_lt_m_w_counts.npy', all_lt_m_w_counts)
    np.save(f'{result_path}/all_times.npy', all_times)
    
def _transform_time_series_to_dial_plot(time_series, slices_per_day, padding=0):
    '''Transforms ascending time series vector to bottom-to-top intra_day (y) and left-to-right inter_day (x) matrix'''
    time_series = np.pad(time_series, (padding, padding), mode='constant', constant_values=np.nan)
    dial_plot = time_series.reshape(-1, slices_per_day) # time is ascending left->right (intra-day) and top->bottom (inter-day)
    dial_plot = dial_plot.T # time ascending left->right (inter-day) and top->bottom (intra-day)
    dial_plot = dial_plot[::-1] # horizontally flip the image so time is moving from bottom to top
    return dial_plot


def print_dial_plot(in_path, aggregation_interval_minutes=10, shift_by_fraction_of_day = 0.5, dest='data/output/dial_plot.pdf'):
    """
    Generates and saves a dial plot for the data generated by `compute_count_over_years`. Plot is saved to `dest`.
    
    The plot can be shifted by a fraction of a day along the y-axis to be able to place interesting features in the center of the plot 
    and not have them be cut off by the transition to the next day.

    Parameters:
    in_path (str): The input path where the numpy files 'all_lt_m_w_counts.npy' and 'all_times.npy' are located.
    aggregation_interval_minutes (int, optional): The interval in minutes for aggregating the data. 
                                                  Needs to be the same as used in `compute_count_over_years`. Default is 10 minutes.
    shift_by_fraction_of_day (float, optional): The fraction of a day to shift the plot along the y-axis. Default is 0.5 (half a day).
    dest (str, optional): The destination path where the dial plot will be saved. Default is 'data/output/dial_plot.pdf'.
    """
    all_lt_m_w_counts:np.ndarray = np.load(f'{in_path}/all_lt_m_w_counts.npy')
    all_times = np.load(f'{in_path}/all_times.npy')

    slices_per_day = 24 * 60 // aggregation_interval_minutes
    padding = int(slices_per_day*shift_by_fraction_of_day)# pad fraction of a day in the beggining and end to shift along day time (y) scale.

    # convert unix to datetime and get date string 
    first_day = datetime.datetime.fromtimestamp(min(all_times))
    last_day = datetime.datetime.fromtimestamp(max(all_times))

    dial_plots = [_transform_time_series_to_dial_plot(all_lt_m_w_counts[:, i], slices_per_day, padding=padding) for i in range(all_lt_m_w_counts.shape[1])]
    time = _transform_time_series_to_dial_plot(all_times, slices_per_day, padding=padding)

    fig, axes = plt.subplots(len(dial_plots), 1, figsize=(12, 5))
    for ax, dial_plot in zip(axes, dial_plots):
        im = ax.imshow(dial_plot,aspect=2)

        # print yearly dates on x-axis (plus last date)
        idx_labels = [(first_day.replace(year=first_day.year + i)).strftime('%Y-%m-%d') for i in range(last_day.year - first_day.year)] 
        idx_labels += [''] + [last_day.strftime('%Y-%m-%d')]
        idx = [i * 365 for i in range(len(idx_labels)-1)] + [dial_plot.shape[1] - 1]
        ax.set_xticks(idx)
        ax.set_xticklabels(idx_labels, fontsize=5)
        ax.set_xticks([min(i * 365//4, dial_plot.shape[1] - 1) for i in range(len(idx_labels)*4)], minor=True)

        # print time in a da on y-axis
        idx = list(range(0, dial_plot.shape[0], int(6/24 * slices_per_day))) + [dial_plot.shape[0] - 1]
        ax.set_yticks(idx)
        time_not_nan_columns = time[:, np.all(~np.isnan(time), axis=0)] # find a column in time which is not none 
        wall_clock = [datetime.datetime.fromtimestamp(t) for t in time_not_nan_columns[idx, 0]]
        # round to next hour
        wall_clock = [(dt + datetime.timedelta(hours=int(round(dt.minute / 60)))).strftime('%Hh') for dt in wall_clock] 
        ax.set_yticklabels(wall_clock, fontsize=5)

        fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
    plt.savefig(dest, bbox_inches='tight', pad_inches=0)
                

def parse_args():
    """Parses args for the CNN inference model."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('configs',          str,   'CNN/configs/fish3.yml',         "configure")
    add_arg("use_gpu",          bool,  True,                        "use gpu")
    add_arg('save_matrix_path', str,   'CNN/output/images/',            "save_matrix_path")
    add_arg('resume_model',     str,   'CNN/models_trained/Res18lstm_MelSpectrogram/best_model',  "model path")
    add_arg('output_txt',     str,   'CNN/outputs/test54_e.txt',  "txt path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    np.random.seed(42)
    args = parse_args()
    cnn_model = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)
    yolo_model = YOLOMultiLabelClassifier("YOLO/final_model/weights",)
    in_path = 'convert_to_wav/wav'
    out_path = 'data/analyzed'

    _infer_model = _infer_yolo # set to _infer_cnn or _infer_yolo
    seg_dur_s = 3 # segment duration in seconds; THIS MUST MATCH YOUR EXISTING DATA

    #----------------------------------------------------------------------#
    # Comment the functions you don't want to run. 
    # Functions starting with _ are not meant to be called directly.
    #----------------------------------------------------------------------#

    # infer_all(in_path, out_path)
    # infer(path=f"{in_path}/20210707", out_path=out_path,skip_existing=True)

    # analyze_by_day('YOLO/labeled_data/train/audio/Montijo_20210712_0.wav', 
    #                aggregation_interval_minutes=1, start_time_s=0, file_ending='.pdf')

    # analyze_all_days_by_day('data/analyzed_subset')

    # compute_count_over_years(out_path, "data")
    # print_dial_plot("data")

    plot_against_validation_data('YOLO/labeled_data/validation/audio', skip_existing=False)

    # plot_over_time("data", use_cached=True, only_every_nth_day=7, smoothing_window=40)