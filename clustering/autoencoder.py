import librosa
import librosa.display
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import os
from torch.utils.data import Dataset, DataLoader, random_split
import glob
from tqdm import tqdm
from torchinfo import summary
import audiomentations as AA
np.random.seed(0)
torch.manual_seed(0)

def extract_mel_spectrogram(librosa_audio, sr, n_mels=64, n_fft=256, hop_length=64):
    mel_spec = librosa.feature.melspectrogram(y=librosa_audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    if mel_spec.shape[-1] % 8 != 0:
        padding = 8 - (mel_spec.shape[-1] % 8)
        mel_spec = np.pad(mel_spec, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to log scale
    return mel_spec_db


class MelSpectrogramAutoencoder(nn.Module):
    def __init__(self, expected_shape, latent_dim=64):
        """expected_shape as [channels, height, width], where height = n_mels, width = time bins. Should be multiples of 8"""
        super(MelSpectrogramAutoencoder, self).__init__()
        if expected_shape[-1] % 8 != 0 or expected_shape[-2] % 8 != 0:
            raise ValueError("Expected shape should be multiples of 8")
        self.expected_shape = expected_shape  # Store expected shape
        self.latent_dim = latent_dim

        pre_latent_height = expected_shape[-2] // 2**3 # 3 Conv2d layers with stride 2
        pre_latent_width = expected_shape[-1] // 2**3
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),  
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * pre_latent_height * pre_latent_width, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * pre_latent_height * pre_latent_width),
            nn.ReLU(),
            nn.Unflatten(1, (64, pre_latent_height, pre_latent_width)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),  
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        if x.shape[-len(self.expected_shape):] != self.expected_shape:
            raise ValueError(f"Input shape mismatch! Expected {self.expected_shape}, but got {x.shape[1:]}")

        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z  # Return both the reconstructed spectrogram and latent representation

def pad_too_short(spectrogram, expected_width):
    if spectrogram.shape[-1] < expected_width: # pad too short audios
        padding = expected_width - spectrogram.shape[-1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    return spectrogram

class AudioDataset(Dataset):
    def __init__(self, audio_dir, expected_width, sr=4000, n_mels=64, n_fft=256, hop_length=64, min_val=None, max_val=None, should_augment=False, threshold=None):
        if isinstance(audio_dir, str):
            self.audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
        else:
            self.audio_files = audio_dir
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.expected_width = expected_width
        self.should_augment = should_augment
        self.min_val = min_val
        self.max_val = max_val
        self.threshold = threshold

        # Define augmentation pipeline
        self.augment = AA.Compose([
            AA.TimeMask(max_band_part=0.3, p=0.5),
            AA.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
            AA.PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
            AA.Shift(p=0.5, rollover=True),
            AA.Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
            AA.TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=True, p=0.2)
        ])

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.audio_files[idx], sr=self.sr)

        if self.should_augment:
            audio = self.augment(audio, sample_rate=sr)

        mel_spectrogram = extract_mel_spectrogram(audio, sr=sr, 
                                                  n_mels=self.n_mels, hop_length=self.hop_length, n_fft=self.n_fft)
        
        mel_spectrogram = pad_too_short(mel_spectrogram, self.expected_width)
        if self.min_val is not None and self.max_val is not None: # normalize
            mel_spectrogram = (mel_spectrogram - self.min_val) / (self.max_val - self.min_val)

        if self.threshold is not None:
            mel_spectrogram = np.where(mel_spectrogram < self.threshold, 0, mel_spectrogram)
        
        mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0).float()  # Add channel dimension
        return mel_spectrogram  # Shape: (1, 64, 128)

def save_model(model: MelSpectrogramAutoencoder, path, epoch=-1, val_loss=-1):
    model_params = {
        "expected_shape": model.expected_shape,
        "latent_dim": model.latent_dim,
        "epoch": epoch,
        "val_loss": val_loss,
        "state_dict": model.state_dict()
    }

    torch.save(model_params, path)

def load_model(path, device='mps'):
    model_params = torch.load(path, map_location=device)
    model = MelSpectrogramAutoencoder(expected_shape=model_params["expected_shape"], latent_dim=model_params["latent_dim"])
    model.load_state_dict(model_params["state_dict"])
    print(f"Loaded model trained for {model_params['epoch']} epochs with validation loss {model_params['val_loss']}")
    return model

def train(audio_directory, model, device='mps',  num_epochs=100, batch_size=32, lr=0.001, augment=True,
          patience=10, output_dir="clustering/models", starting_epoch=0, num_workers=0, spectro_threshold=None):
    metrics_dataset = AudioDataset(audio_directory, model.expected_shape[-1], should_augment=False)
    metrics_loader = DataLoader(metrics_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    sample = np.array([next(iter(metrics_loader)).numpy() for _ in range(4)])
    min_val, max_val = sample.min(), sample.max()
    print(f"Min value: {min_val}, Max value: {max_val}")

    dataset = AudioDataset(audio_directory, model.expected_shape[-1], min_val=min_val, max_val=max_val, should_augment=augment, threshold=spectro_threshold)
    train_dataset, val_dataset = random_split(dataset, [0.85, 0.15], generator=torch.Generator().manual_seed(0))
    val_dataset.dataset = copy.deepcopy(val_dataset.dataset) 
    val_dataset.dataset.should_augment = False # Disable augmentation for validation set
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize model
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')

    min_val_loss = float("inf")  # Track best validation loss
    best_epoch = 0
    patience_counter = 0  # Count epochs without improvement

    os.makedirs(output_dir, exist_ok=True) 
    # Training loop
    for epoch in tqdm(range(starting_epoch, num_epochs), desc='Epochs', total=num_epochs-starting_epoch):
        model.train()
        train_loss = 0

        for spectrograms in tqdm(train_loader, total=len(train_loader), desc='Train Batches', leave=False):
            spectrograms = spectrograms.to(device)
            
            optimizer.zero_grad()
            recon, _ = model(spectrograms)
            loss = criterion(recon, spectrograms)
            loss.backward()

            # Gradient Clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)

            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for spectrograms in tqdm(val_loader, total=len(val_loader), desc='Validation', leave=False):
                spectrograms = spectrograms.to(device)
                recon, _ = model(spectrograms)
                loss = criterion(recon, spectrograms)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (best in epoch {best_epoch+1}: {min_val_loss:.4f})")

        # Save checkpoint for each epoch
        save_model(model, os.path.join(output_dir, "autoencoder_last.pth"), epoch=epoch, val_loss=val_loss)

        # Save the best model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0  # Reset early stopping counter
            save_model(model, os.path.join(output_dir, "autoencoder_best.pth"), epoch=epoch, val_loss=val_loss)
            print("Best model saved!")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement in epoch {epoch}.")
            break

def check_reconstruction():
    # infer example:
    model = load_model("clustering/models/autoencoder_best.pth").eval()
    example_wav = "YOLO/data/validation/audio_segments/20161117_0757_-08.000-08.167_segment_878.wav"
    audio_set = AudioDataset([example_wav], model.expected_shape[-1], should_augment=False, threshold=0.45, min_val=-80.0, max_val=9.5367431640625e-07)
    mel_spectrogram = next(iter(DataLoader(audio_set, batch_size=1)))
    sr=4000
    with torch.no_grad():
        recon, _ = model(mel_spectrogram)
    recon = recon.squeeze().detach().cpu().numpy()
    print(recon.shape)

    # Plot original and reconstructed spectrogram
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    mel_spectrogram = mel_spectrogram.squeeze().cpu().numpy()
    librosa.display.specshow(mel_spectrogram, sr=sr, hop_length=64, x_axis='time', y_axis='mel', ax=axs[0])
    axs[0].set_title(f"Original Mel Spectrogram {np.min(mel_spectrogram)}, {np.max(mel_spectrogram)}")
    librosa.display.specshow(recon, sr=sr, hop_length=64, x_axis='time', y_axis='mel', ax=axs[1])
    axs[1].set_title(f"Reconstructed Mel Spectrogram {np.min(recon)}, {np.max(recon)}")
    plt.show()

if __name__ == '__main__':
    # --- TRAIN NEW MODEL ---
    # model = MelSpectrogramAutoencoder(expected_shape=(1, 64, 64), latent_dim=64) # 1 Channel, 64 mel bins, 64 time bins (1 s audio at 4000 Hz (rounded to next multiple of 8))
    # batch_size=128
    # in_size = (batch_size,) + model.expected_shape
    # summary(model, input_size=in_size)
    # train("YOLO/data/train/audio_segments", model, device='mps', num_epochs=150, num_workers=0,
    #       batch_size=batch_size , lr=0.001, patience=10, output_dir="clustering/models", augment=True, spectro_threshold=0.45)

    check_reconstruction()
