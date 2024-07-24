#!/usr/bin/env python3

import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from imblearn.over_sampling import SMOTE, RandomOverSampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/input/data')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--output-data-dir', type=str, default='/opt/ml/output/data')
    args = parser.parse_args()

    input_data_path = args.input_data
    model_dir = args.model_dir
    output_data_dir = args.output_data_dir
    input_data_file = os.path.join(input_data_path, 'qa_pairs.json')

    print(f"input_data_path: {input_data_path}")
    print(f"model_dir: {model_dir}")
    print(f"output_data_dir: {output_data_dir}")
    print(f"input_data_file: {input_data_file}")

    if not os.path.exists(input_data_file):
        raise FileNotFoundError(f"Training data file not found at {input_data_file}")

    with open(input_data_file, 'r') as f:
        qa_pairs = json.load(f).get('qa_pairs', [])

    print(f"Loaded {len(qa_pairs)} Q/A pairs.")

    # Convert to DataFrame
    data = pd.DataFrame(qa_pairs)
    print(f"DataFrame shape: {data.shape}")
    print(data.head())

    X = data['question']
    y = data['answer']

    print(f"Example question: {X.iloc[0]}")
    print(f"Example answer: {y.iloc[0]}")

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)
    print(f"TF-IDF vectorization complete. Shape: {X_tfidf.shape}")

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Label encoding complete. Example encoded labels: {y_encoded[:5]}")

    # Inspect class distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Class distribution: {class_distribution}")

    # Find the minimum number of samples in any class
    min_samples = min(class_distribution.values())

    # Use SMOTE with k_neighbors set to min_samples - 1 if min_samples > 1, otherwise use RandomOverSampler
    if (min_samples > 1):
        smote = SMOTE(random_state=42, k_neighbors=min_samples - 1)
    else:
        smote = RandomOverSampler(random_state=42)

    X_tfidf_balanced, y_encoded_balanced = smote.fit_resample(X_tfidf, y_encoded)
    print(f"After resampling - X shape: {X_tfidf_balanced.shape}, y shape: {y_encoded_balanced.shape}")

    # Example: simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf_balanced, y_encoded_balanced, test_size=0.2, random_state=42)
    print(f"Train/test split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Example: train a simple model (Logistic Regression in this case)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    print(f"Model training complete.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    # Save the model, vectorizer, and label encoder
    model_path = os.path.join(model_dir, 'model.joblib')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.joblib')
    label_encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Model, vectorizer, and label encoder saved.")

if __name__ == '__main__':
    main()

