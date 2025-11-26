"""
Optional: Advanced training script with command-line arguments
Usage: python train_config.py --epochs 10 --batch-size 16 --lr 0.0001
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import os
import argparse

# Local imports
from src.dataset import MelSpectrogramDataset
from src.models import load_efficientnet_b0


def parse_args():
    parser = argparse.ArgumentParser(description='TVM Audio Deepfake Detection Training')

    # Dataset args
    parser.add_argument('--token', type=str,
                        default='TODO TOKEN',
                        help='HuggingFace token')
    parser.add_argument('--dataset', type=str,
                        default='aurigin/TVM_dataset',
                        help='HuggingFace dataset name')
    parser.add_argument('--cache-dir', type=str,
                        default='./hf_cache',
                        help='Dataset cache directory')

    # Training args
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers')

    # Output args
    parser.add_argument('--output', type=str,
                        default='./submission/submission.csv',
                        help='Output submission file path')
    parser.add_argument('--model-save', type=str,
                        default='best_model.pth',
                        help='Path to save best model')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("TVM Audio Deepfake Detection - Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

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

    # Load dataset
    print("\n" + "=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)

    dataset = load_dataset(
        args.dataset,
        token=args.token,
        cache_dir=args.cache_dir
    )

    print(f"\n[INFO] Dataset loaded!")
    print(f"  - Train: {len(dataset['train'])}")
    print(f"  - Val: {len(dataset['validation'])}")
    print(f"  - Test: {len(dataset['test'])}")

    # Create dataloaders
    train_ds = MelSpectrogramDataset(dataset["train"], train=True)
    val_ds = MelSpectrogramDataset(dataset["validation"], train=True)
    test_ds = MelSpectrogramDataset(dataset["test"], train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Load model
    print("\n" + "=" * 60)
    print("STEP 2: Loading Model")
    print("=" * 60)

    model = load_efficientnet_b0(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"[INFO] Model loaded (lr={args.lr})")

    # Training loop
    print("\n" + "=" * 60)
    print("STEP 3: Training")
    print("=" * 60)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for mel_img, labels in tqdm(train_loader, desc="[TRAIN]"):
            mel_img, labels = mel_img.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(mel_img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for mel_img, labels in tqdm(val_loader, desc="[VAL]  "):
                mel_img, labels = mel_img.to(device), labels.to(device)
                outputs = model(mel_img)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100. * val_correct / val_total

        print(f"\n[SUMMARY] Epoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_save)
            print(f"  ✓ Best model saved! ({val_acc:.2f}%)")

    print(f"\nBest Val Acc: {best_val_acc:.2f}%")

    # Predictions
    print("\n" + "=" * 60)
    print("STEP 4: Generating Predictions")
    print("=" * 60)

    model.load_state_dict(torch.load(args.model_save))
    model.eval()

    test_ids = []
    test_preds = []

    with torch.no_grad():
        for mel_img, ids in tqdm(test_loader, desc="[TEST]"):
            mel_img = mel_img.to(device)
            outputs = model(mel_img)
            _, predicted = outputs.max(1)
            test_ids.extend(ids)
            test_preds.extend(predicted.cpu().numpy().astype(float))

    # Save submission
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    submission_df = pd.DataFrame({"id": test_ids, "label": test_preds})
    submission_df.to_csv(args.output, index=False)

    print(f"\n[INFO] Submission saved to: {args.output}")
    print("\n" + "=" * 60)
    print("DONE! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
