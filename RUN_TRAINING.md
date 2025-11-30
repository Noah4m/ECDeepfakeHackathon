# How to Run Training (No Jupyter!)

## Quick Start - Simple Method

Just run this single command:

```bash
# 1. Activate your virtual environment
source venv/bin/activate

# 2. Run training
python train.py
```

That's it! The script will:
- Load the dataset
- Train for 5 epochs
- Save the best model to `best_model.pth`
- Generate predictions to `submission/submission.csv`

## Output You'll See

```
============================================================
TVM Audio Deepfake Detection - Training
============================================================

[INFO] Using device: cuda
[INFO] GPU: NVIDIA GeForce RTX 3080

============================================================
STEP 1: Loading Dataset
============================================================
[INFO] Loading dataset from HuggingFace: aurigin/TVM_dataset
[INFO] Dataset loaded successfully!
  - Train samples: 7500
  - Validation samples: 1500
  - Test samples: 5000

============================================================
STEP 2: Creating DataLoaders
============================================================
[INFO] DataLoaders created:
  - Train batches: 235
  - Validation batches: 47
  - Test batches: 157

============================================================
STEP 3: Loading Model
============================================================
[INFO] Model loaded successfully!
  - Architecture: EfficientNet-B0
  - Output classes: 2 (Real vs Fake)
  - Optimizer: Adam (lr=0.0001)

============================================================
STEP 4: Training Model
============================================================

============================================================
Epoch 1/5
============================================================
[TRAIN] Epoch 1/5: 100%|████████| 235/235 [02:15<00:00, loss=0.5234, acc=73.21%]
[VAL]   Epoch 1/5: 100%|████████| 47/47 [00:22<00:00, acc=75.67%]

[SUMMARY] Epoch 1:
  Train Loss: 0.5234
  Train Acc:  73.21%
  Val Acc:    75.67%
  ✓ New best model saved! (Val Acc: 75.67%)

...

============================================================
Training Complete!
Best Validation Accuracy: 82.34%
============================================================

============================================================
STEP 5: Generating Test Predictions
============================================================
[TEST] Predicting: 100%|████████| 157/157 [01:23<00:00]
[INFO] Generated 5000 predictions

============================================================
STEP 6: Saving Submission
============================================================
[INFO] Submission saved to: ./submission/submission.csv

First 10 predictions:
        id  label
  test_001    0.0
  test_002    1.0
  test_003    0.0
  ...

============================================================
ALL DONE! 🎉
============================================================

Output files:
  - Best model: best_model.pth
  - Submission: ./submission/submission.csv

You can now submit ./submission/submission.csv to the competition!
============================================================
```

## Advanced Method - Custom Configuration

If you want to change hyperparameters:

```bash
# Run with custom settings
python train_config.py \
  --epochs 10 \
  --batch-size 16 \
  --lr 0.00005 \
  --num-workers 4
```

Available options:
- `--epochs N`: Number of training epochs (default: 5)
- `--batch-size N`: Batch size (default: 32, reduce if out of memory)
- `--lr X`: Learning rate (default: 0.0001)
- `--num-workers N`: Data loading workers (default: 2)
- `--output PATH`: Submission file path
- `--model-save PATH`: Model checkpoint path

## Running in VSCode

### Method 1: Run in Terminal

1. Open VSCode terminal (`Ctrl+` ` or View → Terminal)
2. Run:
   ```bash
   source venv/bin/activate
   python train.py
   ```

### Method 2: Use VSCode Python Runner

1. Open `train.py` in VSCode
2. Click the **▶ Run** button in the top-right corner
3. Or press `F5` to run with debugger

### Method 3: Add to VSCode Tasks

Create `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Train Model",
            "type": "shell",
            "command": "${workspaceFolder}/venv/bin/python",
            "args": ["train.py"],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        }
    ]
}
```

Then run with `Cmd+Shift+B` (macOS) or `Ctrl+Shift+B` (Windows/Linux).

## Files Created After Training

```
ECDeepfakeHackathon/
├── best_model.pth           ← Trained model weights
├── submission/
│   └── submission.csv       ← Your predictions (submit this!)
└── hf_cache/                ← Downloaded dataset
```

## Monitoring Training

The script shows:
- Real-time progress bars for each epoch
- Training loss and accuracy
- Validation accuracy
- Automatically saves best model based on validation accuracy

## Troubleshooting

### Out of Memory Error
Reduce batch size in `train.py`:
```python
BATCH_SIZE = 16  # or 8
```

### Slow Training
- Reduce `NUM_WORKERS` if CPU is slow
- Consider using GPU if available

### Import Errors
Make sure you're in the project root directory:
```bash
pwd
# Should show: .../ECDeepfakeHackathon
```

## What's Different from Jupyter?

✓ **Cleaner output** - Better formatted progress bars and logs
✓ **Saves best model** - Automatically tracks and saves best checkpoint
✓ **Resume-friendly** - Can stop and restart without re-running cells
✓ **VSCode integration** - Run, debug, and edit in one place
✓ **Progress tracking** - Real-time loss and accuracy in progress bars

## Next Steps

After training completes:
1. Check `submission/submission.csv` for predictions
2. Review training logs to see which epoch performed best
3. Submit `submission.csv` to competition
4. Optionally tweak hyperparameters and re-train

Enjoy training! 🚀
