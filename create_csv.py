import pandas as pd
import os
import glob

# Create training CSV
def create_voicebank_csv():
    root = "./voicebank_dns_format"
    
    # Training data
    train_clean_files = sorted(glob.glob(os.path.join(root, "training_set/clean/*.wav")))
    train_noisy_files = sorted(glob.glob(os.path.join(root, "training_set/noisy/*.wav")))
    
    # Create training CSV
    train_df = pd.DataFrame({
        'clean': train_clean_files,
        'noisy': train_noisy_files
    })
    train_df.to_csv("voicebank_train.csv", index=False)
    
    # Test data
    test_clean_files = sorted(glob.glob(os.path.join(root, "datasets/test_set/synthetic/no_reverb/clean/*.wav")))
    test_noisy_files = sorted(glob.glob(os.path.join(root, "datasets/test_set/synthetic/no_reverb/noisy/*.wav")))
    
    # Create test CSV
    test_df = pd.DataFrame({
        'clean': test_clean_files,
        'noisy': test_noisy_files
    })
    test_df.to_csv("voicebank_test.csv", index=False)
    
    print(f"Created training CSV with {len(train_df)} samples")
    print(f"Created test CSV with {len(test_df)} samples")

create_voicebank_csv()
