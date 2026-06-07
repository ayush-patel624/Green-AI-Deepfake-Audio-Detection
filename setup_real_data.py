import os
import shutil
import random

# Configuration
REAL_AUDIO_DIR = "real_audio"
FAKE_AUDIO_DIR = "fake_audio"
OUTPUT_DIR = "my_dataset"

def create_protocol_line(file_id, label):
    # Format: SPEAKER_ID  FILE_ID  ENV  ATTACK_TYPE  LABEL
    return f"SPK_99 {file_id} - - {label}\n"

def main():
    # Ensure input directories exist
    if not os.path.exists(REAL_AUDIO_DIR) or not os.path.exists(FAKE_AUDIO_DIR):
        print(f"Error: Please create '{REAL_AUDIO_DIR}' and '{FAKE_AUDIO_DIR}' folders first!")
        return

    dataset = []

    # Gather Real files and label them 'bonafide'
    for f in os.listdir(REAL_AUDIO_DIR):
        if f.endswith((".wav", ".flac", ".mp3")):
            dataset.append((os.path.join(REAL_AUDIO_DIR, f), f, "bonafide"))

    # Gather Fake files and label them 'spoof'
    for f in os.listdir(FAKE_AUDIO_DIR):
        if f.endswith((".wav", ".flac", ".mp3")):
            dataset.append((os.path.join(FAKE_AUDIO_DIR, f), f, "spoof"))

    total_files = len(dataset)
    if total_files == 0:
        print("No audio files found in the folders!")
        return

    # Shuffle to mix real and fake files randomly across train/dev/eval
    random.shuffle(dataset)

    # Split: 60% Train, 20% Dev, 20% Eval
    train_split = int(total_files * 0.6)
    dev_split = int(total_files * 0.8)

    splits = {
        "train": dataset[:train_split],
        "dev": dataset[train_split:dev_split],
        "eval": dataset[dev_split:]
    }

    # Clear old dataset if it exists
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    # Process and copy files
    for split, items in splits.items():
        audio_dir = os.path.join(OUTPUT_DIR, f"ASVspoof2019_LA_{split}", "flac")
        os.makedirs(audio_dir, exist_ok=True)
        
        protocol_dir = os.path.join(OUTPUT_DIR, "ASVspoof2019_LA_cm_protocols")
        os.makedirs(protocol_dir, exist_ok=True)
        
        protocol_ext = "trn" if split == "train" else "trl"
        protocol_file = os.path.join(protocol_dir, f"ASVspoof2019.LA.cm.{split}.{protocol_ext}.txt")
        
        print(f"Generating {split} split ({len(items)} files)...")
        
        with open(protocol_file, "w") as pf:
            for source_path, filename, label in items:
                # Remove the original extension to get the clean file ID
                file_id = os.path.splitext(filename)[0]
                
                # Write the exact answer key (bonafide or spoof)
                pf.write(create_protocol_line(file_id, label))
                
                # Copy file and save as .flac for pipeline compatibility
                dest_path = os.path.join(audio_dir, file_id + ".flac")
                shutil.copy(source_path, dest_path)
                
    print(f"\nSuccess! Organized {total_files} files with accurate labels into '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()