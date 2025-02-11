import sys
import os
sys.path.append(".")
import analyze
from YOLO.infer_yolo import segment_audios
import umap
import matplotlib.pyplot as plt
import hdbscan
import numpy as np
import glob
import librosa
import torch
from tqdm import tqdm
np.random.seed(0)
torch.manual_seed(0)

base_cache_dir = "data/cache"
device = 'mps'

def cache_results(cache_dir):
    def decorator(func):
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

if __name__ == '__main__':
    paths = glob.glob("YOLO/data/validation/audio_segments/*.wav")

    feature_maps = _infer_cnn(paths)   
    print(feature_maps.shape)

    feature_maps = _infer_panns(paths)
    print(feature_maps.shape)

    umap_2d = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = umap_2d.fit_transform(feature_maps)

    hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=15)  # Adjust as needed
    clusters_hdb = hdbscan_cluster.fit_predict(embedding_2d)

    # Plot 2D UMAP visualization
    plt.figure(figsize=(15, 3.75))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=clusters_hdb, cmap="viridis", alpha=0.7)
    plt.title("UMAP Projection (2D)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()