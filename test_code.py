"""
Quick test script to verify the code works before running the full notebook
"""
import sys
import torch
import numpy as np

print("Testing imports...")
try:
    from src.dataset import MelSpectrogramDataset
    from src.models import load_efficientnet_b0
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\nTesting model loading...")
try:
    model = load_efficientnet_b0(num_classes=2)
    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"  Output classes: 2")

    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"  Test forward pass: input {dummy_input.shape} -> output {output.shape}")
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
    print("✓ Model forward pass successful")
except Exception as e:
    print(f"✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting dataset with mock data...")
try:
    # Create a mock HuggingFace dataset-like object
    class MockDataset:
        def __init__(self):
            # Simulate audio samples (16kHz, 3 seconds = 48000 samples)
            self.samples = [
                {"audio_array": np.random.randn(48000).astype(np.float32), "label": 0, "id": "test_001"},
                {"audio_array": np.random.randn(48000).astype(np.float32), "label": 1, "id": "test_002"},
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    mock_hf_split = MockDataset()

    # Test training mode
    train_dataset = MelSpectrogramDataset(mock_hf_split, train=True)
    print(f"✓ Dataset created (train mode): {len(train_dataset)} samples")

    mel_img, label = train_dataset[0]
    print(f"  Sample 0: mel_img shape={mel_img.shape}, label={label}")
    assert mel_img.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {mel_img.shape}"
    assert isinstance(label, torch.Tensor), f"Expected label to be tensor, got {type(label)}"
    print("✓ Training dataset working correctly")

    # Test test mode
    test_dataset = MelSpectrogramDataset(mock_hf_split, train=False)
    mel_img, sample_id = test_dataset[0]
    print(f"  Test mode: mel_img shape={mel_img.shape}, id={sample_id}")
    assert mel_img.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {mel_img.shape}"
    print("✓ Test dataset working correctly")

except Exception as e:
    print(f"✗ Dataset test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("✓ ALL TESTS PASSED!")
print("="*50)
print("\nThe code is working correctly. You can now:")
print("1. Run the Jupyter notebook")
print("2. Or use this as a starting point for your training script")
