"""
predict.py
----------
Test a single audio file against a trained Audio Deepfake Detection model.

Usage:
    python predict.py --audio my_test_file.wav --model outputs/models/SVM_model.pkl
"""

import argparse
import os
import sys
import numpy as np
import librosa
import joblib
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Constants (Must match training setup)
MODEL_ID = "facebook/wav2vec2-base-960h"
TARGET_SR = 16000
LAYER_INDEX = 2  # Matches the layer-index 2 you used in training

def load_audio(file_path):
    """Loads audio and forces it to 16kHz Mono for Wav2Vec 2.0."""
    try:
        # librosa automatically resamples to 16kHz and converts to mono
        speech, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
        return speech
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        sys.exit(1)

def extract_embedding(speech, processor, w2v_model, device):
    """Passes audio through Wav2Vec 2.0 and extracts the Layer 2 embedding."""
    inputs = processor(speech, sampling_rate=TARGET_SR, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = w2v_model(input_values)
        # Get the hidden state for the specified layer
        hidden_states = outputs.hidden_states[LAYER_INDEX]
        # Mean pooling across the time dimension
        embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        
    return embedding.reshape(1, -1) # Reshape for sklearn (1 sample, n_features)

def main():
    parser = argparse.ArgumentParser(description="Test a single audio file for Deepfake Detection.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the .wav or .flac file.")
    parser.add_argument("--model", type=str, default="outputs/models/SVM_model.pkl", help="Path to trained .pkl model.")
    parser.add_argument("--scaler", type=str, default="outputs/models/SVM_scaler.pkl", help="Path to trained .pkl scaler.")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("\n[1/4] Loading audio file...")
    speech = load_audio(args.audio)

    logger.info(f"[2/4] Loading Wav2Vec 2.0 Feature Extractor on {device}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    w2v_model = Wav2Vec2Model.from_pretrained(MODEL_ID, output_hidden_states=True).to(device)
    w2v_model.eval()

    logger.info("[3/4] Extracting acoustic features...")
    embedding = extract_embedding(speech, processor, w2v_model, device)

    logger.info("[4/4] Running Classifier...")
    classifier = joblib.load(args.model)
    
    # Scale features if a scaler exists
    if os.path.exists(args.scaler):
        scaler = joblib.load(args.scaler)
        embedding = scaler.transform(embedding)
    
    # Predict (1 = Bonafide/Real, 0 = Spoof/Fake based on train.py LABEL_MAP)
    prediction = classifier.predict(embedding)[0]
    
    # Get Confidence Score
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(embedding)[0]
        confidence = probabilities[prediction] * 100
    else:
        confidence = 100.0 # Fallback if model doesn't support probability

    result_text = "🟢 REAL (Bonafide)" if prediction == 1 else "🔴 FAKE (Spoof)"
    
    print("\n" + "="*40)
    print(" 🎙️  AUDIO ANALYSIS RESULT  🎙️")
    print("="*40)
    print(f"File:       {os.path.basename(args.audio)}")
    print(f"Prediction: {result_text}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()