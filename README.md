# TVM Audio Deepfake Detection

A Mel-Spectrogram → EfficientNet-B0 pipeline for detecting deepfake audio using the TVM dataset.

## 🚀 Quick Start

```bash
# 1. Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate    # On Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training
python train.py
```

That's it! Your model will train and save predictions to `submission/submission.csv`.

## 📁 Project Structure

```
├── src/
│   ├── dataset.py          # MelSpectrogramDataset class
│   ├── models.py           # EfficientNet-B0 model loader
│   └── __init__.py
├── train.py                # Main training script (use this!)
├── train_config.py         # Training with CLI arguments
├── test_code.py            # Verification script
├── requirements.txt        # Python dependencies
├── RUN_TRAINING.md         # Detailed training guide
├── hf_cache/               # Dataset cache (auto-created)
└── submission/             # Output directory
    └── submission.csv      # Predictions (created after training)
```

## 🎯 What It Does

1. **Loads TVM dataset** from HuggingFace (cached locally)
2. **Converts audio to mel-spectrograms** (128 mel bins → 224×224×3 images)
3. **Trains EfficientNet-B0** pretrained on ImageNet
4. **Saves predictions** in CSV format for submission

## 📊 Training Output

- **Training time**: 10-20 min (GPU) or 1-2 hours (CPU)
- **Target accuracy**: 75-85% validation accuracy
- **Output files**:
  - `best_model.pth` - Best model checkpoint
  - `submission/submission.csv` - Test set predictions

## 🔧 Advanced Usage

Customize hyperparameters with `train_config.py`:

```bash
python train_config.py \
  --epochs 10 \
  --batch-size 16 \
  --lr 0.00005 \
  --num-workers 4
```

## 📖 Documentation

- [RUN_TRAINING.md](RUN_TRAINING.md) - Complete training guide with troubleshooting

## 🧪 Verify Installation

```bash
python test_code.py
```

This tests that all imports work and data processing is correct.

## 🛠️ Tech Stack

- **PyTorch** - Deep learning framework
- **EfficientNet-B0** - Pretrained CNN architecture
- **TorchAudio** - Audio processing (mel-spectrograms)
- **HuggingFace Datasets** - TVM dataset loader

## 📝 Dataset

Using the [TVM Dataset](https://huggingface.co/datasets/aurigin/TVM_dataset):
- **Train**: ~7,500 samples
- **Validation**: ~1,500 samples
- **Test**: ~5,000 samples

Binary classification: 0 = Real audio, 1 = Fake/deepfake audio

## 🎓 Hackathon Ready

This codebase is optimized for hackathons:
- ✅ Clean, runnable scripts (no Jupyter required)
- ✅ Simple one-command training
- ✅ Automatic best model checkpointing
- ✅ Progress bars and real-time metrics
- ✅ Ready-to-submit CSV output

Happy hacking! 🚀