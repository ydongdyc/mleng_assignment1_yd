#!/usr/bin/env python3

import argparse
import os
import sys
from datetime import datetime
from sentence_transformers import SentenceTransformer
from joblib import load


def parse_arguments():
    parser = argparse.ArgumentParser(description="Score headlines with sentiment model")
    parser.add_argument("input_file", help="Text file containing headlines (one per line)")
    parser.add_argument("source", help="Source of the headlines (e.g., nyt, chicagotribune)")
    return parser.parse_args()


def validate_file(file_path):
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)


def load_headlines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def vectorize_headlines(headlines):
    print("Loading embedding model...")
    model_path = "/opt/huggingface_models/all-MiniLM-L6-v2"
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
    else:
        print("Local model path not found. Downloading from HuggingFace...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Vectorizing headlines...")
    return model.encode(headlines)


def load_svm_model(model_path="svm_model.joblib"):
    print("Loading sentiment classifier...")
    return load(model_path)


def predict_sentiments(embeddings, classifier):
    return classifier.predict(embeddings)


def write_output(labels, headlines, source):
    today = datetime.today()
    filename = f"headline_scores_{source}_{today.year}_{today.month:02d}_{today.day:02d}.txt"
    print(f"Writing output to {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        for label, headline in zip(labels, headlines):
            f.write(f"{label},{headline}\n")
    print("Done.")


def main():
    if len(sys.argv) < 3:
        print("\n❌ Error: Missing required arguments.")
        print("Usage: python score_headlines.py <input_file> <source>")
        print("Example: python score_headlines.py todaysheadlines.txt nyt")
        print("\nYou must provide both:")
        print("  1️⃣ An input text file with one headline per line")
        print("  2️⃣ A source name (e.g., nyt, chicagotribune)")
        sys.exit(1)

    args = parse_arguments()

    validate_file(args.input_file)
    headlines = load_headlines(args.input_file)
    embeddings = vectorize_headlines(headlines)
    classifier = load_svm_model()
    labels = predict_sentiments(embeddings, classifier)
    write_output(labels, headlines, args.source)


if __name__ == "__main__":
    main()
