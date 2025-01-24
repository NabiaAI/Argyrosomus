import os
import numpy as np
import pydub
import pandas as pd
from tqdm import tqdm

src_dir = 'dat'
dest_dir = 'wav'
skip_existing = True
file_info = pd.read_csv('file_info.csv')

# fine_info.csv example:
# start_date,end_date,channels,sampling_rate(Hz),size_of_files_(kb),note
# 20160422,20160427,2,6000,168.758,firstrecords
assert len(file_info) > 0, "file_info.csv is empty."

def gather_files(src_dir):
    files = []
    for root, _, filenames in os.walk(src_dir, onerror=lambda e: print(e)):
        for filename in filenames:
            if filename.endswith('.dat'):
                relative_root = os.path.relpath(root, src_dir)
                files.append((src_dir, os.path.join(relative_root, filename)))
    return sorted(files)

def get_info(filename: str):
    date = filename.split('/')[-2] # get directory name (date)
    assert date.isdigit() and len(date) == 8, f"Expecting YYYYMMDD. Date format is incorrect: {date}"

    date = int(date)
    row = file_info[(file_info.start_date <= date) & (file_info.end_date >= date)]
    if row.empty:
        return None, None
    assert len(row) == 1, f"Expecting one row for date {date}. Found {len(row)} rows (ambigious)."
    sr, ch = row['sampling_rate(Hz)'].values[0], row['channels'].values[0]
    if sr < 0 or ch < 0: # invalid data
        return None, None
    return sr, ch

def convert_to_wav(src, dest, orig_sampling_rate, numch, target_sampling_rate=4000):
    with open(src, "rb") as f:
        valorFile = 4096  # header of the dat files
        f.seek(valorFile * np.dtype(np.int16).itemsize, os.SEEK_SET) # skip header

        data = np.fromfile(f, dtype=np.int16)
        channel_0 = data[::numch] # only get first channel
        channel_0 -= 32768

    audio = pydub.AudioSegment(data=channel_0.tobytes(), sample_width=2, frame_rate=orig_sampling_rate, channels=1)
    resampled = audio.set_frame_rate(target_sampling_rate)
    resampled.export(dest, format="wav")

if __name__ == '__main__':
    files = gather_files(src_dir)
    for root, file in tqdm(files):
        sampling_rate, numch = get_info(file)
        if sampling_rate is None and numch is None:
            print(f"Skipping {file}. Sampling rate and number of channels not found or invalid in file_info.csv.")
            continue

        dest = os.path.join(dest_dir, file.replace('.dat', '.wav'))
        if skip_existing and os.path.exists(dest):
            print(f"Skipping {file}. File already exists.")
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        convert_to_wav(os.path.join(root, file), dest, sampling_rate, numch)
