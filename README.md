# Linguistic Steganalysis Using Semantic and Statistical Feature Fusion

A hybrid text steganalysis framework that combines ModernBERT semantic embeddings with statistical TF-IDF features to detect hidden information embedded in natural language text.

The framework integrates contextual semantic representations from ModernBERT with compressed statistical representations derived from TF-IDF and AutoEncoder networks. The fused representation is then used to classify text as either cover text or steganographic text.

---

## Overview

Text steganography hides secret information within seemingly natural text. Recent generation-based steganographic techniques such as RNN-Stega and VAE-Stega produce highly fluent text, making detection increasingly challenging.

This project proposes a hybrid steganalysis framework that combines:

- ModernBERT semantic embeddings
- TF-IDF statistical features
- AutoEncoder-based dimensionality reduction
- Feature fusion for binary classification

The model was evaluated on the SeSy benchmark dataset across multiple datasets, coding schemes, and steganographic techniques.

---

## Architecture

<p align="center">
  <img src="images/architecture.png" alt="Model Architecture" width="800"/>
</p>

*Overall architecture of the proposed linguistic steganalysis framework.*

---

## Dataset

The project utilizes the dataset released by the SeSy framework.

### Source Corpora

- IMDB Reviews
- News Articles
- Twitter Posts

### Steganographic Models

- RNN-Stega
- VAE-Stega

### Coding Schemes

- Huffman Coding (HC)
- Arithmetic Coding (AC)

Each dataset configuration contains:

- 10,000 cover samples
- 10,000 stego samples

for a total of 20,000 balanced samples.

---

## Repository Structure

```text
.
├── Code/
│   ├── ablation.py
│   ├── main.py
│
├── Dataset/
│   ├── RNN_Stega
│   ├── VAE_Stega
│
└── README.md
```

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- ModernBERT
- Scikit-Learn
- Pandas

---

## Experimental Configuration

| Parameter | Value |
|------------|------------|
| Optimizer | AdamW |
| Learning Rate | 1e-5 |
| Batch Size | 16 |
| Dropout | 0.2 |
| Epochs | 10 |
| Early Stopping | Patience = 3 |

---

## Results

### Best F1 Scores

| Dataset | RNN-Stega (HC) | VAE-Stega (HC) |
|----------|----------|----------|
| IMDB | 96.00 | 96.92 |
| News | 98.34 | 98.57 |
| Twitter | 93.33 | 93.65 |

### Ablation Study

| Configuration | IMDB F1 (HC) |
|--------------|--------------|
| ModernBERT Only | 86.15 |
| ModernBERT + AutoEncoded TF-IDF | 96.00 |

The ablation study demonstrates that integrating statistical features with ModernBERT substantially improves classification performance compared to using semantic representations alone.

---

## Key Contributions

- Hybrid semantic and statistical feature fusion framework.
- Integration of ModernBERT and AutoEncoded TF-IDF representations.
- Competitive performance against state-of-the-art SeSy baseline models.
- Strong performance across multiple datasets and encoding schemes.
- Demonstrates the value of combining semantic and statistical information for text steganalysis.

---

## How to Run

### 1. Clone Repository

```bash
git clone https://github.com/HardiMatholia/Linguistic_Steganalysis.git
```

### 2. Change directory
```bash
cd Linguistic_Steganalysis
```

### 3. Setup Instructions

1. **Update Dataset Paths**  
   - Modify the paths to the dataset files in the script to match the dataset locations on your machine.

2. **Install Dependencies**  
   - Make sure all required Python packages are installed 

### 4. Train and Evaluate Model

```bash
python main.py
```
---

✅ *Happy Training!* 😊  
