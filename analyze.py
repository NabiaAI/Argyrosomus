import sys
import os
import shutil
import h5py
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
    
def infer_cnn(args, audios, sample_rate):
    audios = [AudioSegment.from_ndarray(audio, sample_rate) for audio in audios]
    trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)
    outs, preds = trainer.predict(audios,resume_model=args.resume_model)
    return preds,outs

def infer_yolo(audios, sample_rate):
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
    return np.array(preds), boxes, box_preds_batch

def save_array_list(path, arrays: list[np.ndarray]):
    arrays_with_idx = []
    for i, array in enumerate(arrays):
        if array.size == 0:
            continue
        arrays_with_idx.append(np.hstack([np.ones((array.shape[0], 1)) * i, array]))
    array_with_idx = np.vstack(arrays_with_idx, dtype=np.float32)
    np.savez_compressed(path, boxes_with_segment_idx=array_with_idx)

def infer(args, path: str, out_path, skip_existing=True, save_box_preds=False):
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
            return
    paths = sorted(paths)
    
    out_times, out_preds, out_boxes, out_bpreds = f'{out_path}/{name}_times.npz', f'{out_path}/{name}_preds.npz', f'{out_path}/{name}_boxes.npz', f'{out_path}/{name}_boxpreds.npy'
    if os.path.exists(out_times) and os.path.exists(out_preds) and skip_existing:
        print(f"Skipping {name} as output files already exist")
        return

    print("processing", name)
    times, audios, sample_rate = segment_audios(paths)

    #preds, boxes = infer_cnn(args, audios, sample_rate)
    preds, boxes, box_preds = infer_yolo(audios, sample_rate)
    save_array_list(out_boxes, boxes)
    np.savez_compressed(out_times, times=times)
    np.savez_compressed(out_preds, preds=preds)
    if save_box_preds:
        np.save(out_bpreds, box_preds)

def _get_full_path_of_audio_files(audio_path):
    audio_files = {}
    for root, _, files in os.walk(audio_path):
        for file in files:
            if not file.lower().endswith('.wav'):
                continue
            assert file not in audio_files, f"Duplicate file {file}"
            audio_files[file] = os.path.join(root, file)
    return audio_files

def smooth_and_plot(x, y, window_size, label, color, only_moveing_average=False):
    # Compute the moving average
    y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    x_smooth = np.convolve(x, np.ones(window_size)/window_size, mode='valid')

    if only_moveing_average:
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

def plot_analysis(all_ratios, all_m_w_counts, n_boot, batch_number, slope, slope_ci, intercept, r_value, trend_str, path):
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
    smooth_and_plot(plt_ratios[:,0], plt_lt_m_w[:,0], window_size, None, 'orange')
    smooth_and_plot(plt_ratios[:,0], plt_lt_m_w[:,1], window_size, None, 'green')
    smooth_and_plot(plt_ratios[:,0], plt_lt_m_w[:,2], window_size, None, 'red')

    ax2.set_ylabel('Counts'), ax2.legend(fontsize='small')
    plt.grid(axis='x', alpha=0.2)

    #plt.xticks(np.arange(batch_number), np.arange(batch_number) + 2015)
    CI_str = f"(95%CI: [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}])"if n_boot > 1 else ""
    #plt.text(0.03, 0.83, f'r: {r_value:.3f} (r²: {r_value**2:.3f}), slope {slope:.3f} {CI_str}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    #plt.text(0.03, 0.79, trend_str, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.savefig(f'{path}_daily.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

def analyze(in_path, n_boot=1, ):
    print(f"Analyzing {in_path}")
    preds = np.load(f'{in_path}_preds.npz')['preds'] # shape (n, 3)
    times = np.load(f'{in_path}_times.npz')['times'] # shape (n,) in seconds of the day
    valid_times = (0 <= times) & (times < 23 * 3600 + 59 * 60 + 59) # only keep times within a day
    preds = preds[valid_times]
    times = times[valid_times]

    aggregation_interval_minutes = 10
    batch_number = 24*60 // aggregation_interval_minutes
    batch_size = len(preds) // batch_number
    all_ratios = []
    all_lt_m_w_counts = []
    for i in range(batch_number):
        batch = preds[i * batch_size:(i + 1) * batch_size]
        time = times[i * batch_size:(i + 1) * batch_size].mean() / 3600 # in hours
        lt_m_w_counts, ratios = quant.bootstrap(preds=batch, which_ratio=[1,2], n=n_boot)
        batch_ratios = np.vstack([np.ones_like(ratios) * time, ratios]).T
        all_ratios.append(batch_ratios)
        all_lt_m_w_counts.append(lt_m_w_counts)

    slopes = []
    ratios_3d = np.array(all_ratios)
    for i in range(0, n_boot):
        slope, _, _, _, _ = linregress(ratios_3d[:,i,0], ratios_3d[:,i,1])
        slopes.append(slope)
    
    mannkendall = pymannkendall.original_test([r[:,1].mean() for r in all_ratios])
    print(f"Mann-Kendall monotony test: trend: {mannkendall.trend}, tau={mannkendall.Tau:.3f}, p-value={mannkendall.p:.3f}, slope={mannkendall.slope:.3f}")

    all_ratios = np.vstack(all_ratios)
    all_lt_m_w_counts = np.vstack(all_lt_m_w_counts)
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

    plot_analysis(all_ratios, all_lt_m_w_counts, n_boot, batch_number, slope, slope_ci, intercept, r_value, trend_str, path=in_path)
    return all_lt_m_w_counts

def infer_all(in_path, out_path, skip_existing=True):
    for root, dirs, _ in os.walk(in_path):
        dirs = sorted(dirs)
        for d in tqdm(dirs):
            p = os.path.join(root, d)
            infer(args, path=p, out_path=out_path, skip_existing=skip_existing)

def get_true_counts(path_to_selection_table):
    label_idx_mapping = {
        'lt': 0,
        'm': 1,
        'w': 2
    }
    df = pd.read_csv(path_to_selection_table, sep='\t')
    if len(df) == 0:
        return np.array([0,0,0]), np.array([0,0,0])

    labels = []
    for start_time in range(0, int(df['End Time (s)'].max())+10, 5): # loop over segments of 5 seconds
        end_time = start_time + 5
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

def analyze_all(in_path, load_labels=False):
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
                counts = analyze(p)
                dates.append(f)
                sums.append(counts.sum(axis=0)/len(counts)) # average over time slots (negates the effect if a day has less data)
    sums = np.array(sums)
    dates = np.array(dates)
    return dates, sums

def analyze_against_validation_data(validation_path, skip_existing=False):
    dest = f"data/validation_analyzed"
    for f in sorted(os.listdir(validation_path)):
        p = os.path.join(validation_path, f)
        if f.endswith('.wav'):
            infer(args, path=p, out_path=dest, skip_existing=skip_existing, save_box_preds=True)
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
            segment_counts, total_counts = get_true_counts(raven_file)
            true_counts.append(segment_counts), true_counts_total.append(total_counts)
            dates.append(f)
    
    sums, true_counts = np.array(sums), np.array(true_counts)
    box_preds_sums, true_counts_total = np.array(box_preds_sums), np.array(true_counts_total)
    dates = list(map(lambda x: f"{x[15:17]}h-{x[6:8]}.{x[4:6]}.{x[2:4]}", dates))

    eval.plot_validation_output(dates, sums, true_counts, save_path=f"{dest}/result_segment.pdf")
    eval.plot_validation_output(dates, box_preds_sums, true_counts_total, save_path=f"{dest}/result_total_calls.pdf")


def plot_over_time(in_path):
    sums, dates = analyze_all(in_path)
    dates = list(map(lambda x: x[:4] + '.' + x[4:6] + '.' + x[6:8], dates))

    # plot the sums against the dates
    plt.figure()
    plt.plot(dates, sums[:,0], label='lt', marker="o", color='orange')
    plt.plot(dates, sums[:,1], label='m', marker="x", color='green')
    plt.plot(dates, sums[:,2], label='w', marker="+", color='red')
    plt.xticks(np.arange(len(dates)), dates, rotation=90)
    plt.legend()
    plt.savefig(f'{in_path}/over_years.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('configs',          str,   'CNN/configs/fish3.yml',         "Configs")
    add_arg("use_gpu",          bool,  True,                        "USe GPU or not")
    add_arg('resume_model',     str,   'CNN/models//Res18_MelSpectrogram/best_model/',  "model path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    np.random.seed(42)
    args = parse_args()
    yolo_model = YOLOMultiLabelClassifier("YOLO/final_model/weights",)
    in_path = 'convert_to_wav/wav'
    out_path = 'data/analyzed'
    infer_all(in_path, out_path)
    #infer(args, path=f"{in_path}/20210707", out_path=out_path,skip_existing=True)

    analyze_all(out_path)

    # analyze_against_validation_data('YOLO/data/validation/audio', skip_existing=True)