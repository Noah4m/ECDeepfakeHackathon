"""
TVM Audio Deepfake Detection - Training Script
Run this script directly: python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import os

# Local imports
from src.dataset import MelSpectrogramDataset
from src.models import load_efficientnet_b0

# ============================================================================
# CONFIGURATION
# ============================================================================

HF_TOKEN = "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"
DATASET_NAME = "aurigin/TVM_dataset"
CACHE_DIR = "./hf_cache"
SUBMISSION_PATH = "./submission/submission.csv"

BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 2  # Adjust based on your CPU cores

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    print("=" * 60)
    print("TVM Audio Deepfake Detection - Training")
    print("=" * 60)

    # Check device - prioritize MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n[INFO] Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n[INFO] Using device: {device}")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"\n[INFO] Using device: {device} (Warning: Training will be slow)")

    # ========================================================================
    # 1. LOAD DATASET
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)

    print(f"[INFO] Loading dataset from HuggingFace: {DATASET_NAME}")
    print(f"[INFO] Cache directory: {CACHE_DIR}")

    dataset = load_dataset(
        DATASET_NAME,
        token=HF_TOKEN,
        cache_dir=CACHE_DIR
    )

    print(f"\n[INFO] Dataset loaded successfully!")
    print(f"  - Train samples: {len(dataset['train'])}")
    print(f"  - Validation samples: {len(dataset['validation'])}")
    print(f"  - Test samples: {len(dataset['test'])}")

    # ========================================================================
    # 2. CREATE DATALOADERS
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Creating DataLoaders")
    print("=" * 60)

    train_ds = MelSpectrogramDataset(dataset["train"], train=True)
    val_ds = MelSpectrogramDataset(dataset["validation"], train=True)
    test_ds = MelSpectrogramDataset(dataset["test"], train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"[INFO] DataLoaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # ========================================================================
    # 3. LOAD MODEL
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Loading Model")
    print("=" * 60)

    print("[INFO] Loading EfficientNet-B0 with pretrained weights...")
    model = load_efficientnet_b0(num_classes=2).to(device)

    # Label smoothing helps prevent overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # LR scheduler reduces learning rate when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    print(f"[INFO] Model loaded successfully!")
    print(f"  - Architecture: EfficientNet-B0")
    print(f"  - Output classes: 2 (Real vs Fake)")
    print(f"  - Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  - Label smoothing: 0.1")
    print(f"  - LR scheduler: ReduceLROnPlateau")

    # ========================================================================
    # 4. TRAINING LOOP
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Training Model")
    print("=" * 60)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"[TRAIN] Epoch {epoch+1}/{NUM_EPOCHS}")
        for mel_img, labels in train_pbar:
            mel_img = mel_img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(mel_img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # ---- VALIDATION ----
        model.eval()
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc=f"[VAL]   Epoch {epoch+1}/{NUM_EPOCHS}")
        with torch.no_grad():
            for mel_img, labels in val_pbar:
                mel_img = mel_img.to(device)
                labels = labels.to(device)

                outputs = model(mel_img)
                _, predicted = outputs.max(1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress bar
                val_pbar.set_postfix({
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })

        val_acc = 100. * val_correct / val_total

        # Print epoch summary
        print(f"\n[SUMMARY] Epoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Acc:  {train_acc:.2f}%")
        print(f"  Val Acc:    {val_acc:.2f}%")

        # Update learning rate based on validation loss
        scheduler.step(avg_train_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")

    # ========================================================================
    # 5. PREDICTION ON TEST SET
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Generating Test Predictions")
    print("=" * 60)

    # Load best model
    print("[INFO] Loading best model for predictions...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_ids = []
    test_preds = []

    test_pbar = tqdm(test_loader, desc="[TEST] Predicting")
    with torch.no_grad():
        for mel_img, ids in test_pbar:
            mel_img = mel_img.to(device)

            outputs = model(mel_img)
            _, predicted = outputs.max(1)

            test_ids.extend(ids)
            test_preds.extend(predicted.cpu().numpy().astype(float))

    print(f"[INFO] Generated {len(test_preds)} predictions")

    # ========================================================================
    # 6. SAVE SUBMISSION
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Saving Submission")
    print("=" * 60)

    submission_df = pd.DataFrame({
        "id": test_ids,
        "label": test_preds
    })

    # Create submission directory if it doesn't exist
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"[INFO] Submission saved to: {SUBMISSION_PATH}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10).to_string(index=False))

    # ========================================================================
    # DONE
    # ========================================================================
    print("\n" + "=" * 60)
    print("ALL DONE! 🎉")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Best model: best_model.pth")
    print(f"  - Submission: {SUBMISSION_PATH}")
    print(f"\nYou can now submit {SUBMISSION_PATH} to the competition!")
    print("=" * 60)


if __name__ == "__main__":
    main()
