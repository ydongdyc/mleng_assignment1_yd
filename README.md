# Headline Sentiment Scorer

This project contains a Python script to classify news headlines as **Optimistic**, **Pessimistic**, or **Neutral** using a pre-trained SVM sentiment model and SentenceTransformer embeddings.

## 📄 Overview

The main script, `score_headlines.py`, accepts a text file of headlines and classifies each one using an SVM model trained on vectorized headlines. It outputs the results in a timestamped file that can be used to monitor daily sentiment trends.

## 🛠 How to Use

Usage: python score_headlines.py <input_file> <source>

Example: python score_headlines.py todaysheadlines.txt nyt

### ⚙️ Requirements

- Python 3.8+
- `sentence-transformers`
- `scikit-learn`
- `joblib`

Install dependencies:

```bash
pip install sentence-transformers scikit-learn joblib
