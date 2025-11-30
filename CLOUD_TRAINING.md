# Cloud Training Options & GPU Acceleration

## 🍎 MacBook GPU (MPS) - UPDATED! ✅

**Good news**: I just updated your scripts to use your MacBook's GPU automatically!

### What Changed:
Your MacBook Air (if M1/M2/M3) has a powerful GPU. The code now detects and uses it via **Metal Performance Shaders (MPS)**.

```python
# Now automatically uses:
# MPS (Apple GPU) > CUDA (NVIDIA) > CPU
device = torch.device("mps")  # Apple Silicon GPU
```

### Expected Speedup:
- **Before (CPU)**: 1-2 hours for 5 epochs
- **After (MPS/GPU)**: 15-30 minutes for 5 epochs
- **Speedup**: ~3-5x faster! 🚀

### To Verify:
When you run `python train.py`, you should see:
```
[INFO] Using device: mps (Apple Silicon GPU)
```

If you see this, you're using the GPU! If it says `cpu`, your Mac might not have Apple Silicon.

---

## ☁️ Free Cloud Training Options

If your MacBook doesn't have Apple Silicon (Intel Mac) or MPS isn't working, here are free cloud options:

### 1. **Google Colab** (BEST FREE OPTION)

**Free GPU**: NVIDIA T4 (16GB VRAM)
**Cost**: FREE (with limitations)
**Training time**: 10-15 minutes

#### How to Use:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook
3. Enable GPU: `Runtime → Change runtime type → GPU`
4. Upload your code or clone from GitHub:

```python
# In Colab cell 1: Clone your repo
!git clone https://github.com/yourusername/ECDeepfakeHackathon.git
%cd ECDeepfakeHackathon

# In Colab cell 2: Install dependencies
!pip install -q torch torchvision torchaudio datasets librosa soundfile pandas tqdm

# In Colab cell 3: Run training
!python train.py
```

5. Download results:
```python
from google.colab import files
files.download('submission/submission.csv')
files.download('best_model.pth')
```

**Limitations**:
- 12-hour session timeout (plenty for 5 epochs)
- Disconnect after ~90 min of inactivity
- GPU not always available during peak hours

**Pro Tip**: Colab Pro ($10/month) gives priority GPU access and longer sessions

---

### 2. **Kaggle Notebooks**

**Free GPU**: NVIDIA P100 or T4
**Cost**: FREE
**Training time**: 10-15 minutes
**Weekly limit**: 30 hours GPU/week

#### How to Use:
1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Create new notebook
3. Enable GPU: `Settings → Accelerator → GPU T4 x2`
4. Add dataset or upload code
5. Run cells similar to Colab

**Advantages over Colab**:
- More reliable GPU availability
- 30 hours/week is generous
- Integrated with Kaggle competitions

**Great for hackathons!**

---

### 3. **Lightning.ai** (formerly Grid.ai)

**Free GPU**: Various (including A100 credits)
**Cost**: FREE tier available
**Training time**: 5-10 minutes

#### How to Use:
1. Sign up at [lightning.ai](https://lightning.ai)
2. Create new Studio
3. Upload code or connect GitHub
4. Select GPU instance
5. Run training

**Advantages**:
- Professional-grade GPUs
- Easy collaboration
- Good for team hackathons

---

### 4. **Paperspace Gradient**

**Free GPU**: Various options
**Cost**: FREE tier with limited hours
**Training time**: 10-15 minutes

Free tier gives 6 hours/month of GPU time.

---

### 5. **Hugging Face Spaces** (Limited)

**Free GPU**: Community GPUs
**Cost**: FREE
**Training time**: Variable

Good for demos, not ideal for training.

---

## 💰 Paid Options (Very Cheap for Short Training)

### 1. **Lambda Labs**
- **Cost**: ~$0.50-1.00/hour (NVIDIA RTX 4090)
- **Total cost for this project**: ~$0.25-0.50
- **Speed**: VERY FAST (5-7 minutes for 5 epochs)

### 2. **RunPod**
- **Cost**: ~$0.30-0.80/hour
- **Total cost**: ~$0.15-0.40
- **Speed**: Fast

### 3. **Vast.ai** (CHEAPEST)
- **Cost**: ~$0.10-0.40/hour
- **Total cost**: ~$0.05-0.20
- **Speed**: Varies by GPU

**For a hackathon, spending $0.50 for 10x speed might be worth it!**

---

## 📊 Understanding Accuracy

### What the Accuracy Score Means

When you see:
```
Val Acc: 78.32%
```

**This means**:
- The model correctly classified **78.32%** of validation samples
- Out of 1,500 validation samples, it got ~1,175 correct
- It misclassified ~325 samples

### Accuracy Formula:
```python
accuracy = (correct_predictions / total_samples) * 100

# Example:
# Correct: 1,175
# Total: 1,500
# Accuracy: (1,175 / 1,500) * 100 = 78.32%
```

### What's a Good Score?

For audio deepfake detection:
- **50%**: Random guessing (coin flip)
- **60-70%**: Model is learning something
- **75-85%**: Good performance ✓ (your target)
- **85-95%**: Excellent performance
- **95%+**: Exceptional (or overfitting)

### Why Validation Accuracy Matters

**Training Accuracy vs Validation Accuracy**:

```
Epoch 1:
  Train Acc: 73.21%   ← How well it fits training data
  Val Acc:   75.67%   ← How well it generalizes to new data
```

**Validation accuracy is more important** because:
1. It shows how well your model works on *unseen* data
2. Test set performance will be close to validation accuracy
3. Prevents overfitting (memorizing training data)

### Reading the Progress Bar

```
[TRAIN] Epoch 1/5: 100%|████| 235/235 [02:15<00:00, loss=0.52, acc=73.21%]
                                                       ^^^^^^   ^^^^^^^^^^^
                                                       Loss     Accuracy
```

- **Loss**: How "wrong" the model is (lower is better)
  - `0.52` = moderately confident, some errors
  - `0.1` = very confident, few errors
  - `2.0` = very confused, many errors

- **Accuracy**: Percentage correct (higher is better)
  - Real-time tracking during training
  - Updates every batch

### What Happens Over Epochs:

```
Epoch 1: Train Loss=0.52 | Val Acc=75.67%
Epoch 2: Train Loss=0.41 | Val Acc=78.12%  ← Improving!
Epoch 3: Train Loss=0.35 | Val Acc=79.45%  ← Still improving
Epoch 4: Train Loss=0.29 | Val Acc=79.38%  ← Plateau
Epoch 5: Train Loss=0.24 | Val Acc=78.92%  ← Might be overfitting
```

**Best model saved**: Epoch 3 (79.45% validation accuracy)

---

## 🎯 Quick Decision Guide

**If you have M1/M2/M3 MacBook**:
→ Use the updated `train.py` with MPS (FREE, 15-30 min)

**If you have Intel MacBook**:
→ Use Google Colab (FREE, 10-15 min)

**If Colab is slow/unavailable**:
→ Try Kaggle Notebooks (FREE, very reliable)

**If you need FAST results for hackathon**:
→ Vast.ai or Lambda Labs ($0.10-0.50 total)

**For learning/experimenting**:
→ Run locally on CPU overnight (FREE, slow but works)

---

## 🚀 Recommended: Google Colab Setup

Here's a complete Colab notebook you can copy-paste:

```python
# Cell 1: Setup
!git clone https://github.com/yourusername/ECDeepfakeHackathon.git
%cd ECDeepfakeHackathon
!pip install -q torch torchvision torchaudio datasets librosa soundfile pandas tqdm

# Cell 2: Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 3: Train
!python train.py

# Cell 4: Download results
from google.colab import files
files.download('submission/submission.csv')
files.download('best_model.pth')
```

**Total time**: ~15 minutes (including setup)
**Cost**: FREE

Good luck with your hackathon! 🎉