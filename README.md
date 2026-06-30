# ML Text Classification using Sparse Features and Contextual Embeddings

## Overview

This project compares traditional sparse text representations with modern contextual embeddings for multi-class text classification. The objective is to evaluate the effectiveness of different feature extraction techniques when used with classical machine learning algorithms.

The project implements multiple feature extraction methods, applies dimensionality reduction techniques, and compares model performance using standard evaluation metrics.

---

## Objectives

- Compare sparse and dense text representations for text classification.
- Evaluate classical machine learning algorithms across multiple feature representations.
- Analyze the impact of dimensionality reduction techniques.
- Compare contextual embeddings with traditional NLP features.

---

## Features

- Text preprocessing pipeline
- Bag of Words (BoW)
- TF-IDF Vectorization
- GloVe Word Embeddings
- DistilBERT Contextual Embeddings
- Principal Component Analysis (PCA)
- Truncated Singular Value Decomposition (SVD)
- Performance visualization and comparison

---

## Machine Learning Models

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## Technologies Used

- Python
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- NumPy
- Pandas
- NLTK
- Matplotlib
- SciPy

---

## Repository Structure

```
.
├── Plot/
├── main.py
├── preprocessing.py
├── features.py
├── models.py
├── evaluation.py
├── pca.py
├── requirements.txt
├── BA2025013_ML_PROJECT_REPORT.pdf
├── For_code_run_in_colab.pdf
└── README.md
```

---

## Project Workflow

```
Raw Text
    │
    ▼
Text Preprocessing
    │
    ▼
Feature Extraction
├── Bag of Words
├── TF-IDF
├── GloVe
└── DistilBERT
    │
    ▼
Model Training
├── Logistic Regression
├── SVM
└── KNN
    │
    ▼
Model Evaluation
├── Accuracy
├── Macro F1-score
└── Performance Comparison
```

---

## Results

The experimental analysis demonstrates that contextual embeddings generated using **DistilBERT** outperform traditional sparse representations such as TF-IDF and Bag of Words for text classification tasks.

Key observations include:

- Improved classification performance using contextual embeddings.
- Effective dimensionality reduction using PCA and Truncated SVD.
- Comparative evaluation of multiple machine learning algorithms on different feature representations.

---

## How to Run

### Clone the repository

```bash
git clone https://github.com/divyansh-laddha/ML-Text-classification-sparse-vs-contextual-embeddings.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python main.py
```

---

## Project Report

The complete project report is available in:

```
BA2025013_ML_PROJECT_REPORT.pdf
```

---

## Authors

- **Divyansh Laddha**
- **BA2025013**

**International Institute of Information Technology Bangalore (IIIT Bangalore)**

---

## Acknowledgements

This project was completed as part of the Machine Learning coursework at **International Institute of Information Technology Bangalore (IIIT Bangalore)**.

---

## License

This repository is intended for academic and educational purposes.
