import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def create_pairs(clean_dir, noisy_dir, ext='wav'):
    clean_files = glob.glob(os.path.join(clean_dir, f'*.{ext}'))
    noisy_files = glob.glob(os.path.join(noisy_dir, f'*.{ext}'))

    clean_dict = {os.path.basename(f): f for f in clean_files}
    noisy_dict = {os.path.basename(f): f for f in noisy_files}

    pairs = []
    missing_noisy, missing_clean = [], []

    for fname, clean_path in clean_dict.items():
        noisy_path = noisy_dict.get(fname)
        if noisy_path:
            pairs.append((noisy_path, clean_path))  # Note: noisy first
        else:
            missing_noisy.append(fname)

    for fname in noisy_dict:
        if fname not in clean_dict:
            missing_clean.append(fname)

    df_pairs = pd.DataFrame(pairs, columns=['noisy', 'clean'])

    print(f" Found pairs: {len(pairs)}")
    if missing_noisy:
        print(f" Missing noisy files for {len(missing_noisy)} clean files: {missing_noisy}")
    if missing_clean:
        print(f" Missing clean files for {len(missing_clean)} noisy files: {missing_clean}")

    return df_pairs

def main():
    clean_dir = 'data/clean'
    noisy_dir = 'data/noisy'
    df = create_pairs(clean_dir, noisy_dir)

    # Split into train/val (90/10)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # Save to CSV
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train_pairs.csv", index=False)
    val_df.to_csv("data/val_pairs.csv", index=False)

    print(f" Saved {len(train_df)} training pairs and {len(val_df)} validation pairs.")

if __name__ == "__main__":
    main()