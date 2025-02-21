import sys
import os
sys.path.append(".")
import analyze
from YOLO.infer_yolo import segment_audios
from YOLO.create_data_yolo import save_spectrogram
import umap
import matplotlib.pyplot as plt
import hdbscan
import numpy as np
import glob
import librosa
import functools
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

base_cache_dir = "data/cache"
device = 'mps'

def cache_results(cache_dir):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, f"{func.__name__}_cache.npy")
            if os.path.exists(cache_file):
                return np.load(cache_file)
            
            results = func(*args, **kwargs)
            
            np.save(cache_file, results)
            return results
        return wrapper
    return decorator

@cache_results(base_cache_dir)
def _infer_panns(paths, batch_size=256) -> np.ndarray:
    import panns_inference
    panns = panns_inference.AudioTagging(checkpoint_path=None, device=device)
    all_embeddings = []

    for i in tqdm(range(0, len(paths), batch_size)):
        batch_paths = paths[i:i + batch_size]
        audio = [librosa.core.load(path, sr=32000, mono=True)[0] for path in batch_paths]
        audio = np.vstack(audio)  # (batch_size, segment_samples)

        (_, embedding) = panns.inference(audio)
        all_embeddings.append(embedding)

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings

@cache_results(base_cache_dir)
def _infer_cnn(paths):
    from CNN.macls.trainer_mlabel import MAClsTrainer

    analyze.args = analyze.parse_args()
    analyze.cnn_model = MAClsTrainer(configs=analyze.args.configs, use_gpu=analyze.args.use_gpu)

    _, audios, sample_rate = segment_audios(paths, extract_timestamps=False)
    _, _, feature_maps = analyze._infer_cnn(audios, sample_rate,)
    return feature_maps


@cache_results(base_cache_dir)
def _infer_autoencoder(paths, batch_size=256, sr=4000):
    from autoencoder import load_model, extract_mel_spectrogram, pad_too_short
    model = load_model("clustering/models/autoencoder_best.pth").eval().to(device)
    expected_width = model.expected_shape[-1]

    all_embeddings = []
    for i in tqdm(range(0, len(paths), batch_size)):
        batch_paths = paths[i:i + batch_size]

        audio = [librosa.core.load(path, sr=sr, mono=True)[0] for path in batch_paths]
        mel_spectrograms = [extract_mel_spectrogram(a, sr=sr) for a in audio]
        mel_spectrograms = np.array([pad_too_short(m, expected_width) for m in mel_spectrograms])
        
        # normalize mel spectrograms
        mel_spectrograms_norm = (mel_spectrograms - mel_spectrograms.min()) / (mel_spectrograms.max() - mel_spectrograms.min())
        # print("median, average, 75 percentile", np.median(mel_spectrograms_norm), np.mean(mel_spectrograms_norm), np.percentile(mel_spectrograms_norm, 75))
        # # distribution of values
        # plt.figure(figsize=(3,3))
        # plt.hist(mel_spectrograms.flatten(), bins=20)
        # plt.show()

        threshold = 0.6
        mel_spectrograms_norm[mel_spectrograms_norm < threshold] = 0
        # plt.imshow(mel_spectrograms_norm[0])
        # plt.show()
        mel_spectrograms_norm = torch.tensor(mel_spectrograms_norm).float().unsqueeze(1).to(device) # add channel dimension

        with torch.no_grad():
            embeddings: torch.tensor = model.encoder(mel_spectrograms_norm)

        all_embeddings.append(embeddings.cpu().numpy()) 

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings

def save_spectrograms_to_clusters(paths, clusters, base_dest, embeddings, sr=4000):
    """
    Reads .wav files and saves their spectrograms to folders based on their cluster labels.

    Parameters:
    paths (list of str): List of paths to .wav files.
    clusters (np.ndarray): Array of cluster labels for each file.
    base_path (str): Base path where all cluster folders are stored.
    sr (int, optional): Sample rate for reading audio files. Default is 4000.
    """
    # normalize embeddings
    embeddings = (embeddings - embeddings.min(axis=0)) / (embeddings.max(axis=0) - embeddings.min(axis=0))

    for path, cluster, embedding in tqdm(zip(paths, clusters, embeddings), total=len(paths), desc="Spectrograms"):
        cluster_dir = os.path.join(base_dest, str(cluster))
        os.makedirs(cluster_dir, exist_ok=True)

        audio, sr = librosa.load(path, sr=sr)
        save_spectrogram(audio, sr, file_name=os.path.basename(path) + "real", image_folder=cluster_dir, frequency_limit_Hz=sr//4)

        # save normalized thresholded mel spectrograms
        from autoencoder import extract_mel_spectrogram
        mel_spectrograms = extract_mel_spectrogram(audio, sr=sr)
        mel_spectrograms_norm = (mel_spectrograms - (-80)) / (1.91e-6 - (-80))
        threshold = 0.6
        mel_spectrograms_norm[mel_spectrograms_norm < threshold] = 0
        mel_spectrograms_norm=mel_spectrograms_norm[::-1] # fix for photo
        plt.imsave(os.path.join(cluster_dir, f"{os.path.splitext(os.path.basename(path))[0]}.png"), mel_spectrograms_norm, cmap='viridis')

        # save embeddings
        # plt.imshow(embedding.reshape(1, -1), aspect='auto', cmap='viridis')
        # plt.colorbar()
        # plt.savefig(os.path.join(cluster_dir, f"{os.path.splitext(os.path.basename(path))[0]}_embedding.png"))
        # plt.close()


def cluster_with_configuration(paths, feature_extractor, step_0_clust=None, step_1_proj=None, step_2_clust=None, base_dest=None, plot=True, fraction=1.0):
    embeddings = feature_extractor(paths)
    assert fraction > 0.0 and fraction <= 1.0
    if fraction < 1.0:
        idx = np.random.choice(len(embeddings), int(fraction * len(embeddings)), replace=False)
        idx.sort()
        embeddings = embeddings[idx]
        paths = [paths[i] for i in idx]

    print("Embeddings shape:", embeddings.shape)

    if step_0_clust:
        clusters = step_0_clust(embeddings)
    if step_1_proj:
        embeddings = step_1_proj(embeddings)
    if step_2_clust:
        clusters = step_2_clust(embeddings)

    print("Number of clusters:", len(set(clusters)))

    functions = [feature_extractor, step_0_clust, step_1_proj, step_2_clust]
    if not base_dest:
        base_dest = os.path.join("data/clustering", *[f.__name__ for f in functions if f])
    
    print("Saving results to", base_dest)
    save_spectrograms_to_clusters(paths, clusters, base_dest, embeddings)

    if not plot: 
        return
    
    clusters = np.array(clusters)
    noise_points = clusters == -1
    if embeddings.shape[1] == 2:
        plt.figure(figsize=(15, 3.75))
        plt.scatter(embeddings[noise_points, 0], embeddings[noise_points, 1], c='red', marker='x', label='Noise')
        plt.scatter(embeddings[~noise_points, 0], embeddings[~noise_points, 1], c=clusters[~noise_points], cmap="viridis", alpha=0.5)
        plt.title("Projection (2D)")
    elif embeddings.shape[1] == 3:
        fig = plt.figure(figsize=(15, 3.75))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings[noise_points, 0], embeddings[noise_points, 1], embeddings[noise_points, 2], c='red', marker='x', label='Noise')
        ax.scatter(embeddings[~noise_points, 0], embeddings[~noise_points, 1], embeddings[~noise_points, 2], c=clusters[~noise_points], cmap="viridis", alpha=0.5)
        ax.set_title("Projection (3D)")
    else:
        print("Cannot plot embeddings with more than 3 dimensions")
    plt.show()

def umap_3d(x):
    return umap.UMAP(n_components=3, random_state=42).fit_transform(x)

def hdb(x):
    return hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(x)

def k_means(x):
    return KMeans(n_clusters=10, random_state=0).fit_predict(x) # boat, silence, toadfish, meagre, weakfish, chorus, noise = 7 clusters?

def k_nn_clustering(x):
    idx = -3
    audio, sr = librosa.load(paths[idx], sr=4000)
    reference_spectro = save_spectrogram(audio, sr, as_array=True)
    plt.imshow(reference_spectro)
    plt.show()

    reference_vector = x[idx]
    distances = np.linalg.norm(x - reference_vector, axis=1)
    sorted_idx = np.argsort(distances)
    rank_positions = np.argsort(sorted_idx)

    clusters = rank_positions // 200 # 200 elements in each cluster
    return clusters

if __name__ == '__main__':
    paths = glob.glob("YOLO/data/validation/audio_segments/*.wav")
    fraction = 0.07

    cluster_with_configuration(paths, _infer_autoencoder, step_0_clust=k_means, plot=False, fraction=fraction)
    cluster_with_configuration(paths, _infer_autoencoder, step_0_clust=hdb, plot=False, fraction=fraction)
    cluster_with_configuration(paths, _infer_autoencoder, step_1_proj=umap_3d, step_2_clust=hdb, plot=False, fraction=fraction)

    # cluster_with_configuration(paths, _infer_cnn, step_0_clust=k_nn_clustering, plot=False, fraction=fraction)
    # cluster_with_configuration(paths, _infer_cnn, step_0_clust=k_means, plot=False, fraction=fraction)

    # cluster_with_configuration(paths, _infer_autoencoder, step_1_proj=umap_3d, step_2_clust=hdb, plot=False, fraction=fraction)
    # cluster_with_configuration(paths, _infer_autoencoder, step_0_clust=hdb, step_1_proj=umap_3d, plot=False, fraction=fraction)

    # cluster_with_configuration(paths, _infer_cnn, step_1_proj=umap_3d, step_2_clust=hdb, plot=False, fraction=fraction)
    # cluster_with_configuration(paths, _infer_cnn, step_1_proj=umap_3d, step_2_clust=k_means, plot=False, fraction=fraction)
    # cluster_with_configuration(paths, _infer_cnn, step_0_clust=k_means, plot=False, fraction=fraction)

    # cluster_with_configuration(paths, _infer_panns, step_1_proj=umap_3d, step_2_clust=hdb, plot=False, fraction=fraction)
    # cluster_with_configuration(paths, _infer_panns, step_0_clust=hdb, plot=False, fraction=fraction)
