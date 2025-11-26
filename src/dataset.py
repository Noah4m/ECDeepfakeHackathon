import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

# Mel-spectrogram extractor (same config everywhere)
mel_extractor = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=128
)

amp_to_db = torchaudio.transforms.AmplitudeToDB()

class MelSpectrogramDataset(Dataset):
    """
    Wraps HuggingFace dataset split and produces:
    - 3×224×224 mel-spectrogram tensors for EfficientNet
    - labels (train/val) or ids (test)
    """

    def __init__(self, hf_split, train=True):
        self.data = hf_split
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # --- Load audio ---
        audio = sample["audio_array"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        else:
            audio = audio.float()

        # --- Mel spectrogram ---
        mel = mel_extractor(audio)
        mel_db = amp_to_db(mel)

        # --- Normalize ---
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        # --- Resize to 224 × 224 and convert to 3 channels ---
        # Add batch dimension: [128, T] -> [1, 1, 128, T]
        mel_db = mel_db.unsqueeze(0).unsqueeze(0)

        # Resize: [1, 1, 128, T] -> [1, 1, 224, 224]
        mel_resized = F.interpolate(
            mel_db,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        # Remove batch dimension and repeat to 3 channels: [1, 1, 224, 224] -> [3, 224, 224]
        mel_img = mel_resized.squeeze(0).repeat(3, 1, 1)

        if self.train:
            label = torch.tensor(int(sample["label"]))
            return mel_img, label
        else:
            return mel_img, sample["id"]
