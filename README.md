# 🎙️ Audio Deepfake Detection via SSL Embeddings 🌱

> **A lightweight, CPU-friendly pipeline for detecting synthetic speech using frozen Wav2Vec 2.0 representations and classical machine learning.**

**Author:** Ayush Patel — *Indian Institute of Information Technology Guwahati (IIITG)*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

Modern audio deepfake detectors rely heavily on large neural networks, GPU clusters, and days of fine-tuning — an approach that is costly, carbon-intensive, and inaccessible to most researchers and practitioners. This project introduces a **resource-efficient, CPU-trainable** detection framework built around three core ideas:

1. **Freeze the foundation model.** Wav2Vec 2.0 BASE is used purely as a feature extractor — no gradients, no fine-tuning, no GPU required for the downstream step.
2. **Use early transformer representations.** The early transformer layers (like layer index 2) produce the most discriminative embeddings for deepfake detection, significantly reducing the computational overhead.
3. **Classical ML at the backend.** Support Vector Machine (SVM), Logistic Regression, XGBoost, and other classifiers replace heavyweight neural decoders, keeping the trainable parameter count extremely low.

## 📊 Key Results

Benchmarked on the **ASVspoof 2019 LA Dataset**, this "Green AI" approach achieves highly competitive results while requiring a fraction of the compute power:
* **Equal Error Rate (EER):** `0.90%`
* **Trainable Parameters:** `< 1,000` (for downstream classifiers)

---

## 🗂️ Project Structure

```text
audio-deepfake-detection/
│
├── extract_features.py      # Extracts Wav2Vec 2.0 embeddings from audio files
├── train.py                 # Trains classical ML classifiers on the embeddings
├── evaluate.py              # Evaluates the model (Computes EER, F1, Accuracy)
├── run_pipeline.py          # End-to-end CLI tool for Extraction → Training → Evaluation
├── predict.py               # Inference script to test a single .wav or .flac file
├── setup_real_data.py       # Helper script to organize custom dataset folders
├── flop_count.py            # Utility to compute GMACs and parameter counts
└── requirements.txt         # Python dependencies

🚀 Getting Started
1. Installation
Clone this repository and install the required dependencies inside an isolated virtual environment.

Bash
git clone [https://github.com/YourUsername/Audio-Deepfake-Detection.git](https://github.com/YourUsername/Audio-Deepfake-Detection.git)
cd Audio-Deepfake-Detection

# Create and activate virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
2. Prepare Your Custom Dataset
Create two folders in the root directory: real_audio/ and fake_audio/. Place your .wav or .flac files into their respective folders.

Run the setup script to automatically organize these into the ASVspoof protocol format required by the pipeline:

Bash
python setup_real_data.py
(This generates a my_dataset/ directory containing train, dev, and eval splits alongside their respective protocol text files).

3. Run the End-to-End Pipeline
You can run the entire process (Feature Extraction -> Model Training -> Evaluation) with a single command:

Bash
python run_pipeline.py \
    --data-root ./my_dataset \
    --output-dir ./outputs \
    --layer-index 2 \
    --model SVM \
    --device cpu
Once finished, check the outputs/results/ folder for your JSON metrics (Accuracy, F1, EER) and generated ROC/DET evaluation plots.

4. Test a Single Audio File
To test a brand-new audio file against your trained model in real-time, use the prediction script:

Bash
python predict.py --audio my_test_file.wav --model outputs/models/SVM_model.pkl
Output Example:

Plaintext
========================================
 🎙️  AUDIO ANALYSIS RESULT  🎙️
========================================
File:       my_test_file.wav
Prediction: 🔴 FAKE (Spoof)
Confidence: 98.45%
========================================
❓ Frequently Asked Questions
Can I use a different Wav2Vec 2.0 model? Yes. You can modify the MODEL_ID in extract_features.py to point to any HuggingFace model compatible with Wav2Vec2Model. Adjust the --layer-index accordingly.

I want to skip re-extraction if CSVs already exist. Use run_pipeline.py --skip-extraction to jump straight to training using the existing embedding CSV files.

Can I run grid search in parallel? Yes. The --n-jobs argument in the pipeline handles parallel workers. It defaults to -1 (uses all available CPU cores).

📜 Credits & Citation
The core methodology and benchmarking results implemented in this codebase are based on the original research:

Saha, S., Sahidullah, M., & Das, S. (2024). Exploring Green AI for Audio Deepfake Detection. arXiv preprint arXiv:2403.14290.

If you use this specific code implementation in your project, feel free to link back to this repository!

⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.